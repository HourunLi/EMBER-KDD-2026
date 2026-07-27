"""GCN-based strict SHOT for binary graph node classification.

Run source training and target adaptation as separate processes.  The target
stage accepts no source data path and does not load the target label or
sensitive attribute until all unsupervised adaptation updates have completed.
"""

from __future__ import print_function

import argparse
import copy
import json
import math
import os
import random

import torch
import torch.nn.functional as functional

from baila_data import (
    compute_feature_statistics,
    load_graph_inputs,
    load_label_column,
    load_target_evaluation_columns,
    load_target_evaluation_files,
    load_value_file,
    standardize_features,
)
from gcn import build_gcn_shot_models, forward_model


CLASS_NUM = 2


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device {} was requested, but CUDA is unavailable".format(
                device_name
            )
        )
    return torch.device(device_name)


def ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def validate_binary_tensor(values, name):
    unique_values = sorted(set(values.detach().cpu().tolist()))
    if unique_values != [0, 1]:
        raise ValueError(
            "{} must contain exactly binary values [0, 1], found {}".format(
                name, unique_values
            )
        )


def stratified_masks(labels, train_ratio, validation_ratio, seed):
    if train_ratio <= 0.0 or validation_ratio <= 0.0:
        raise ValueError("train_ratio and validation_ratio must be positive")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    generator = torch.Generator()
    generator.manual_seed(seed)
    node_count = labels.numel()
    train_mask = torch.zeros(node_count, dtype=torch.bool)
    validation_mask = torch.zeros(node_count, dtype=torch.bool)
    test_mask = torch.zeros(node_count, dtype=torch.bool)

    for class_index in range(CLASS_NUM):
        indices = torch.nonzero(labels == class_index).view(-1)
        if indices.numel() < 3:
            raise ValueError(
                "Class {} needs at least 3 source nodes, found {}".format(
                    class_index, indices.numel()
                )
            )
        order = torch.randperm(indices.numel(), generator=generator)
        indices = indices[order]

        train_count = max(1, int(indices.numel() * train_ratio))
        validation_count = max(1, int(indices.numel() * validation_ratio))
        if train_count + validation_count >= indices.numel():
            validation_count = 1
            train_count = indices.numel() - 2

        train_indices = indices[:train_count]
        validation_indices = indices[
            train_count : train_count + validation_count
        ]
        test_indices = indices[train_count + validation_count :]

        train_mask[train_indices] = True
        validation_mask[validation_indices] = True
        test_mask[test_indices] = True

    return train_mask, validation_mask, test_mask


def label_smoothing_cross_entropy(logits, labels, epsilon):
    class_num = logits.size(1)
    log_probabilities = functional.log_softmax(logits, dim=1)
    with torch.no_grad():
        smoothed = torch.zeros_like(log_probabilities)
        smoothed.fill_(epsilon / class_num)
        smoothed.scatter_(
            1,
            labels.view(-1, 1),
            1.0 - epsilon + epsilon / class_num,
        )
    return torch.mean(torch.sum(-smoothed * log_probabilities, dim=1))


def entropy(probabilities, epsilon=1e-5):
    return torch.sum(
        -probabilities * torch.log(probabilities + epsilon), dim=1
    )


def inverse_learning_rate(optimizer, step, total_steps, gamma=10.0, power=0.75):
    decay = (1.0 + gamma * float(step) / float(max(total_steps, 1))) ** (-power)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = parameter_group["lr0"] * decay
        parameter_group["weight_decay"] = 1e-3
        parameter_group["momentum"] = 0.9
        parameter_group["nesterov"] = True


def create_optimizer(parameter_groups):
    optimizer = torch.optim.SGD(
        parameter_groups,
        momentum=0.9,
        weight_decay=1e-3,
        nesterov=True,
    )
    for parameter_group in optimizer.param_groups:
        parameter_group["lr0"] = parameter_group["lr"]
    return optimizer


def binary_roc_auc(labels, positive_scores):
    """Compute binary ROC-AUC with average ranks for tied scores."""

    labels = labels.detach().cpu().long()
    positive_scores = positive_scores.detach().cpu().float()
    if labels.numel() != positive_scores.numel():
        raise ValueError("Labels and ROC-AUC scores have different lengths")

    positive_count = int(torch.sum(labels == 1).item())
    negative_count = int(torch.sum(labels == 0).item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels")

    ordered = sorted(
        zip(positive_scores.tolist(), labels.tolist()), key=lambda item: item[0]
    )
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1

        # Positions start at zero; statistical ranks start at one.
        average_rank = ((start + 1) + end) / 2.0
        tied_positive_count = sum(
            1 for _, label in ordered[start:end] if label == 1
        )
        positive_rank_sum += average_rank * tied_positive_count
        start = end

    minimum_positive_rank_sum = positive_count * (positive_count + 1) / 2.0
    return (
        positive_rank_sum - minimum_positive_rank_sum
    ) / float(positive_count * negative_count)


def classification_metrics(labels, predictions, positive_scores=None):
    labels = labels.detach().cpu().long()
    predictions = predictions.detach().cpu().long()
    if labels.numel() != predictions.numel():
        raise ValueError("Labels and predictions have different lengths")

    accuracy = torch.mean((labels == predictions).float()).item()
    per_class = []
    recalls = []
    f1_scores = []

    for class_index in range(CLASS_NUM):
        true_positive = torch.sum(
            (labels == class_index) & (predictions == class_index)
        ).item()
        false_positive = torch.sum(
            (labels != class_index) & (predictions == class_index)
        ).item()
        false_negative = torch.sum(
            (labels == class_index) & (predictions != class_index)
        ).item()
        support = torch.sum(labels == class_index).item()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            float(true_positive) / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        recall = (
            float(true_positive) / recall_denominator
            if recall_denominator > 0
            else 0.0
        )
        f1_denominator = precision + recall
        f1 = (
            2.0 * precision * recall / f1_denominator
            if f1_denominator > 0.0
            else 0.0
        )

        recalls.append(recall)
        f1_scores.append(f1)
        per_class.append(
            {
                "class": class_index,
                "support": int(support),
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results = {
        "overall_accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / CLASS_NUM,
        "macro_f1": sum(f1_scores) / CLASS_NUM,
        "per_class": per_class,
    }
    if positive_scores is not None:
        results["roc_auc"] = binary_roc_auc(labels, positive_scores)
        per_target_class_auc = []
        for class_index in range(CLASS_NUM):
            one_vs_rest_labels = (labels == class_index).long()
            class_scores = (
                positive_scores
                if class_index == 1
                else 1.0 - positive_scores
            )
            class_auc = binary_roc_auc(one_vs_rest_labels, class_scores)
            per_class[class_index]["roc_auc"] = class_auc
            per_target_class_auc.append(class_auc)
        results["macro_target_roc_auc"] = (
            sum(per_target_class_auc) / CLASS_NUM
        )
    return results


def fairness_metrics(
    labels, predictions, sensitive, sensitive_name="sensitive_attribute"
):
    labels = labels.detach().cpu().long()
    predictions = predictions.detach().cpu().long()
    sensitive = sensitive.detach().cpu().long()
    if labels.numel() != sensitive.numel():
        raise ValueError("Labels and sensitive attributes have different lengths")
    validate_binary_tensor(sensitive, sensitive_name)

    positive_rates = {}
    true_positive_rates = {}
    false_positive_rates = {}
    group_sizes = {}

    for group_index in (0, 1):
        group_mask = sensitive == group_index
        group_sizes[str(group_index)] = int(torch.sum(group_mask).item())
        positive_rates[str(group_index)] = torch.mean(
            (predictions[group_mask] == 1).float()
        ).item()

        positive_label_mask = group_mask & (labels == 1)
        negative_label_mask = group_mask & (labels == 0)
        if torch.sum(positive_label_mask).item() == 0:
            true_positive_rate = float("nan")
        else:
            true_positive_rate = torch.mean(
                (predictions[positive_label_mask] == 1).float()
            ).item()
        if torch.sum(negative_label_mask).item() == 0:
            false_positive_rate = 0.0
        else:
            false_positive_rate = torch.mean(
                (predictions[negative_label_mask] == 1).float()
            ).item()

        true_positive_rates[str(group_index)] = true_positive_rate
        false_positive_rates[str(group_index)] = false_positive_rate

    demographic_parity_difference = abs(
        positive_rates["0"] - positive_rates["1"]
    )
    equal_opportunity_difference = abs(
        true_positive_rates["0"] - true_positive_rates["1"]
    )
    false_positive_rate_difference = abs(
        false_positive_rates["0"] - false_positive_rates["1"]
    )

    return {
        "group_sizes": group_sizes,
        "positive_prediction_rate": positive_rates,
        "true_positive_rate": true_positive_rates,
        "false_positive_rate": false_positive_rates,
        "demographic_parity_difference": demographic_parity_difference,
        "equal_opportunity_difference": equal_opportunity_difference,
        "equalized_odds_difference": max(
            equal_opportunity_difference, false_positive_rate_difference
        ),
    }


def evaluate_predictions(
    labels,
    predictions,
    positive_scores=None,
    sensitive=None,
    sensitive_name="sensitive_attribute",
):
    results = {
        "classification": classification_metrics(
            labels, predictions, positive_scores=positive_scores
        )
    }
    if sensitive is not None:
        results["fairness"] = fairness_metrics(
            labels,
            predictions,
            sensitive,
            sensitive_name=sensitive_name,
        )
    return results


def headline_metrics(results):
    """Return the four ``learn(1).py`` metrics in percentage points.

    ``accuracy`` is the macro mean of per-target-class accuracies (balanced
    accuracy for this binary task). ``AUC_ROC`` is the macro mean of the
    one-vs-rest target-class AUCs. ``dp`` and ``eo`` are absolute demographic
    parity and equal-opportunity gaps computed from hard predictions. This
    matches the reference evaluator, including its multiplication by 100.
    """

    return {
        "accuracy": results["classification"]["balanced_accuracy"] * 100.0,
        "AUC_ROC": (
            results["classification"]["macro_target_roc_auc"] * 100.0
        ),
        "dp": (
            results["fairness"]["demographic_parity_difference"] * 100.0
        ),
        "eo": (
            results["fairness"]["equal_opportunity_difference"] * 100.0
        ),
    }


def format_percentage_points(value):
    return "{:.2f}%".format(float(value))


def format_headline_metrics(metrics):
    return {
        name: format_percentage_points(value) for name, value in metrics.items()
    }


def print_headline_metrics(title, metrics):
    print("\n{} (%)".format(title))
    for metric_name in ("accuracy", "AUC_ROC", "dp", "eo"):
        print(
            "{:<10} {:.4f}".format(
                metric_name, metrics[metric_name]
            )
        )


def print_result_block(title, results):
    print("\n{}".format(title))
    print(json.dumps(results, indent=2, sort_keys=True))


def build_models_from_configuration(configuration, device):
    encoder, bottleneck, classifier = build_gcn_shot_models(
        input_dim=int(configuration["input_dim"]),
        hidden_dim=int(configuration["hidden_dim"]),
        encoder_dim=int(configuration["encoder_dim"]),
        bottleneck_dim=int(configuration["bottleneck_dim"]),
        class_num=int(configuration["class_num"]),
        dropout=float(configuration["dropout"]),
    )
    return encoder.to(device), bottleneck.to(device), classifier.to(device)


def get_predictions(encoder, bottleneck, classifier, features, adjacency):
    _, predictions, probabilities = get_predictions_with_representations(
        encoder, bottleneck, classifier, features, adjacency
    )
    return predictions, probabilities


def get_predictions_with_representations(
    encoder, bottleneck, classifier, features, adjacency
):
    encoder.eval()
    bottleneck.eval()
    classifier.eval()
    with torch.no_grad():
        representations, logits = forward_model(
            encoder, bottleneck, classifier, features, adjacency
        )
        probabilities = functional.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    return (
        representations.detach().cpu(),
        predictions.detach().cpu(),
        probabilities.detach().cpu(),
    )


def encode_visualization_groups(predictions, sensitive):
    """Encode (predicted Y, sensitive S) using the visualization convention."""

    predictions = predictions.detach().cpu().long().view(-1)
    sensitive = sensitive.detach().cpu().long().view(-1)
    if predictions.numel() != sensitive.numel():
        raise ValueError(
            "Visualization predictions and sensitive values have different lengths"
        )
    labels = torch.full_like(predictions, -1)
    labels[(predictions == 1) & (sensitive == 0)] = 0
    labels[(predictions == 1) & (sensitive == 1)] = 1
    labels[(predictions == 0) & (sensitive == 0)] = 2
    labels[(predictions == 0) & (sensitive == 1)] = 3
    if torch.any(labels < 0).item() or torch.any(labels > 3).item():
        raise ValueError(
            "Visualization labels must use only groups 0, 1, 2, and 3"
        )
    return labels


def export_target_visualization_embeddings(
    output_dir,
    representations,
    predictions,
    sensitive,
    valid_target_mask,
):
    """Write visualization NPZ files directly into the target output folder."""

    representations = representations.detach().cpu()
    predictions = predictions.detach().cpu().long().view(-1)
    sensitive = sensitive.detach().cpu().long().view(-1)
    valid_target_mask = valid_target_mask.detach().cpu().bool().view(-1)
    if representations.dim() != 2:
        raise ValueError("Target representations must be a 2D tensor")
    if representations.size(0) != predictions.numel():
        raise ValueError("Representations and predictions have different lengths")
    if predictions.numel() != sensitive.numel():
        raise ValueError("Predictions and sensitive values have different lengths")
    if predictions.numel() != valid_target_mask.numel():
        raise ValueError("Target validity mask has a different length")
    if not torch.isfinite(representations).all().item():
        raise ValueError("Target representations contain NaN or Inf")

    selected_representations = representations[valid_target_mask]
    selected_predictions = predictions[valid_target_mask]
    selected_sensitive = sensitive[valid_target_mask]
    visualization_labels = encode_visualization_groups(
        selected_predictions, selected_sensitive
    )
    if selected_representations.size(0) != visualization_labels.numel():
        raise ValueError("Visualization representation/label lengths do not match")

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required to export feat.npz and labels.npz"
        ) from error

    representation_array = selected_representations.numpy().astype(
        np.float32, copy=False
    )
    label_array = visualization_labels.numpy().astype(np.int64, copy=False)
    if not np.isfinite(representation_array).all():
        raise ValueError("Target representations contain NaN or Inf")
    if representation_array.shape[0] != label_array.shape[0]:
        raise ValueError("NPZ representation/label lengths do not match")
    if not np.isin(label_array, np.asarray([0, 1, 2, 3], dtype=np.int64)).all():
        raise ValueError("NPZ labels must contain only 0, 1, 2, and 3")

    ensure_directory(output_dir)
    feat_path = os.path.abspath(os.path.join(output_dir, "feat.npz"))
    labels_path = os.path.abspath(os.path.join(output_dir, "labels.npz"))
    np.savez_compressed(feat_path, representations=representation_array)
    np.savez_compressed(labels_path, labels=label_array)
    return {
        "feat_path": feat_path,
        "labels_path": labels_path,
        "representation_key": "representations",
        "label_key": "labels",
        "representation_shape": list(representation_array.shape),
        "label_shape": list(label_array.shape),
        "label_encoding": {
            "0": "predicted_Y=1, sensitive_S=0",
            "1": "predicted_Y=1, sensitive_S=1",
            "2": "predicted_Y=0, sensitive_S=0",
            "3": "predicted_Y=0, sensitive_S=1",
        },
    }


def build_binary_value_mapping(group_zero, group_one, value_name):
    group_zero = str(group_zero)
    group_one = str(group_one)
    if group_zero == group_one:
        raise ValueError(
            "{} group-zero and group-one values must differ".format(value_name)
        )
    return {group_zero: 0, group_one: 1}


def build_label_protocol(
    negative_label,
    positive_label,
    additional_positive_labels,
    ignored_labels,
    ignore_index,
    value_name,
):
    """Build a binary task mapping while retaining unlabeled graph nodes."""

    if int(ignore_index) in (0, 1):
        raise ValueError("ignore_index must differ from binary class indices 0 and 1")
    mapping = build_binary_value_mapping(
        negative_label, positive_label, value_name
    )
    for raw_value in additional_positive_labels:
        raw_value = str(raw_value)
        if raw_value in mapping and mapping[raw_value] != 1:
            raise ValueError(
                "{} value {} cannot be both negative and positive".format(
                    value_name, raw_value
                )
            )
        mapping[raw_value] = 1

    ignored = []
    for raw_value in ignored_labels:
        raw_value = str(raw_value)
        if raw_value in mapping:
            raise ValueError(
                "{} value {} cannot be both mapped and ignored".format(
                    value_name, raw_value
                )
            )
        if raw_value not in ignored:
            ignored.append(raw_value)
    return mapping, ignored


def train_source(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)

    if args.headerless_features:
        excluded_feature_columns = ()
    else:
        excluded_feature_columns = (
            args.id_column,
            args.label_column,
            args.sensitive_column,
        )
    label_mapping, ignored_label_values = build_label_protocol(
        args.negative_label,
        args.positive_label,
        args.additional_positive_labels,
        args.ignored_labels,
        args.ignore_index,
        args.label_column,
    )
    sensitive_mapping = build_binary_value_mapping(
        args.sensitive_group_zero,
        args.sensitive_group_one,
        args.sensitive_column,
    )

    if excluded_feature_columns:
        print(
            "Loading source graph inputs without {}...".format(
                ", ".join(excluded_feature_columns)
            )
        )
    else:
        print(
            "Loading headerless source model features; labels and sensitive "
            "attributes are stored separately..."
        )
    source_graph = load_graph_inputs(
        args.csv,
        args.edges,
        excluded_feature_columns=excluded_feature_columns,
        id_column=args.id_column if args.edges_use_node_ids else None,
        has_header=not args.headerless_features,
    )
    print(
        "Loading {} separately for supervised source training...".format(
            args.label_column
        )
    )
    if args.label_file is None:
        source_labels = load_label_column(
            args.csv,
            args.label_column,
            value_mapping=label_mapping,
            ignored_values=ignored_label_values,
            ignore_index=args.ignore_index,
        )
    else:
        source_labels = load_value_file(
            args.label_file,
            value_name=args.label_column,
            value_mapping=label_mapping,
            ignored_values=ignored_label_values,
            ignore_index=args.ignore_index,
        )
    if source_graph.num_nodes != source_labels.numel():
        raise ValueError("Source features and labels have different lengths")
    labeled_source_mask = source_labels != args.ignore_index
    if torch.sum(labeled_source_mask).item() == 0:
        raise ValueError("The source graph has no labeled nodes")
    validate_binary_tensor(
        source_labels[labeled_source_mask],
        "labeled source {}".format(args.label_column),
    )
    print(
        "Source task nodes: {} labeled, {} ignored/unlabeled, {} total".format(
            int(torch.sum(labeled_source_mask).item()),
            int(torch.sum(~labeled_source_mask).item()),
            source_graph.num_nodes,
        )
    )

    train_mask, validation_mask, test_mask = stratified_masks(
        source_labels,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    feature_mean, feature_std = compute_feature_statistics(
        source_graph.features, train_mask
    )
    normalized_features = standardize_features(
        source_graph.features, feature_mean, feature_std
    )

    features = normalized_features.to(device)
    adjacency = source_graph.adjacency.to(device)
    labels = source_labels.to(device)
    train_mask_device = train_mask.to(device)

    configuration = {
        "input_dim": int(features.size(1)),
        "hidden_dim": args.hidden_dim,
        "encoder_dim": args.encoder_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "class_num": CLASS_NUM,
        "dropout": args.dropout,
    }
    encoder, bottleneck, classifier = build_models_from_configuration(
        configuration, device
    )

    optimizer = create_optimizer(
        [
            {"params": encoder.parameters(), "lr": args.lr * 0.1},
            {"params": bottleneck.parameters(), "lr": args.lr},
            {"params": classifier.parameters(), "lr": args.lr},
        ]
    )

    best_validation_accuracy = -1.0
    best_states = None
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        bottleneck.train()
        classifier.train()
        inverse_learning_rate(optimizer, epoch - 1, args.epochs)

        _, logits = forward_model(
            encoder, bottleneck, classifier, features, adjacency
        )
        source_loss = label_smoothing_cross_entropy(
            logits[train_mask_device],
            labels[train_mask_device],
            epsilon=args.smooth,
        )

        optimizer.zero_grad()
        source_loss.backward()
        optimizer.step()

        should_evaluate = (
            epoch == 1
            or epoch % args.log_interval == 0
            or epoch == args.epochs
        )
        if should_evaluate:
            predictions, probabilities = get_predictions(
                encoder, bottleneck, classifier, features, adjacency
            )
            validation_results = classification_metrics(
                source_labels[validation_mask],
                predictions[validation_mask],
                positive_scores=probabilities[validation_mask, 1],
            )
            validation_accuracy = validation_results["overall_accuracy"]
            print(
                "Source epoch {:04d}/{:04d} loss={:.6f} val_acc={:.4f} "
                "val_bal_acc={:.4f}".format(
                    epoch,
                    args.epochs,
                    source_loss.item(),
                    validation_accuracy,
                    validation_results["balanced_accuracy"],
                )
            )
            if validation_accuracy >= best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_states = {
                    "encoder": copy.deepcopy(encoder.state_dict()),
                    "bottleneck": copy.deepcopy(bottleneck.state_dict()),
                    "classifier": copy.deepcopy(classifier.state_dict()),
                }

    if best_states is None:
        raise RuntimeError("Source training did not produce a checkpoint")

    encoder.load_state_dict(best_states["encoder"])
    bottleneck.load_state_dict(best_states["bottleneck"])
    classifier.load_state_dict(best_states["classifier"])
    source_predictions, source_probabilities = get_predictions(
        encoder, bottleneck, classifier, features, adjacency
    )
    validation_results = classification_metrics(
        source_labels[validation_mask],
        source_predictions[validation_mask],
        positive_scores=source_probabilities[validation_mask, 1],
    )
    test_results = classification_metrics(
        source_labels[test_mask],
        source_predictions[test_mask],
        positive_scores=source_probabilities[test_mask, 1],
    )

    checkpoint = {
        "format_version": 3,
        "experiment": args.experiment,
        "source_domain": args.source_domain,
        "id_column": args.id_column,
        "label_column": args.label_column,
        "sensitive_column": args.sensitive_column,
        "label_mapping": label_mapping,
        "ignored_label_values": ignored_label_values,
        "ignore_index": args.ignore_index,
        "sensitive_mapping": sensitive_mapping,
        "edges_use_node_ids": args.edges_use_node_ids,
        "align_target_features_by_name": args.align_target_features_by_name,
        "headerless_features": args.headerless_features,
        "separate_evaluation_files": args.label_file is not None,
        "excluded_feature_columns": list(excluded_feature_columns),
        "feature_names": list(source_graph.feature_names),
        "feature_schema": list(source_graph.feature_schema),
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "model_configuration": configuration,
        "encoder_state": encoder.state_dict(),
        "bottleneck_state": bottleneck.state_dict(),
        "classifier_state": classifier.state_dict(),
        "seed": args.seed,
    }
    ensure_parent_directory(args.checkpoint)
    torch.save(checkpoint, args.checkpoint)

    print("\nSaved source-only model artifact to {}".format(args.checkpoint))
    print("Feature columns used by the model: {}".format(source_graph.feature_names))
    print_result_block("Best source validation results", validation_results)
    print_result_block("Source held-out test results", test_results)


def obtain_strict_shot_pseudo_labels(
    encoder,
    bottleneck,
    classifier,
    features,
    adjacency,
    refine_rounds,
    threshold,
    epsilon,
):
    """Original SHOT weighted-centroid then hard-centroid label refinement."""

    encoder.eval()
    bottleneck.eval()
    classifier.eval()
    with torch.no_grad():
        bottleneck_features, logits = forward_model(
            encoder, bottleneck, classifier, features, adjacency
        )
        probabilities = functional.softmax(logits, dim=1)
        initial_predictions = torch.argmax(probabilities, dim=1)

        # The released SHOT implementation appends a constant before cosine
        # normalization; retain that behavior for the strict baseline.
        ones = torch.ones(
            bottleneck_features.size(0),
            1,
            dtype=bottleneck_features.dtype,
            device=bottleneck_features.device,
        )
        normalized_features = functional.normalize(
            torch.cat((bottleneck_features, ones), dim=1), p=2, dim=1
        )
        assignments = initial_predictions
        affiliations = probabilities

        for _ in range(refine_rounds):
            centers = torch.mm(affiliations.t(), normalized_features)
            center_mass = affiliations.sum(dim=0).view(-1, 1)
            centers = centers / (center_mass + epsilon)

            class_counts = torch.bincount(
                assignments, minlength=CLASS_NUM
            ).float()
            label_set = torch.nonzero(class_counts > threshold).view(-1)
            if label_set.numel() == 0:
                raise RuntimeError(
                    "No class survived pseudo-label threshold {}".format(
                        threshold
                    )
                )

            selected_centers = functional.normalize(
                centers[label_set], p=2, dim=1
            )
            cosine_distance = 1.0 - torch.mm(
                normalized_features, selected_centers.t()
            )
            nearest_center = torch.argmin(cosine_distance, dim=1)
            assignments = label_set[nearest_center]

            affiliations = torch.zeros_like(probabilities)
            affiliations.scatter_(1, assignments.view(-1, 1), 1.0)

        final_counts = torch.bincount(
            assignments, minlength=CLASS_NUM
        ).detach().cpu()
        mean_probability = probabilities.mean(dim=0).detach().cpu()

    return assignments.detach(), final_counts, mean_probability


def strict_shot_loss(logits, pseudo_labels, cls_par, ent_par, epsilon):
    pseudo_label_loss = functional.cross_entropy(logits, pseudo_labels)
    probabilities = functional.softmax(logits, dim=1)
    conditional_entropy = torch.mean(entropy(probabilities, epsilon=epsilon))
    mean_probability = probabilities.mean(dim=0)
    marginal_entropy = torch.sum(
        -mean_probability * torch.log(mean_probability + epsilon)
    )
    information_maximization = conditional_entropy - marginal_entropy
    total_loss = (
        cls_par * pseudo_label_loss + ent_par * information_maximization
    )
    components = {
        "pseudo_label": pseudo_label_loss.detach().item(),
        "conditional_entropy": conditional_entropy.detach().item(),
        "marginal_entropy": marginal_entropy.detach().item(),
        "information_maximization": information_maximization.detach().item(),
        "total": total_loss.detach().item(),
    }
    return total_loss, components


def load_source_artifact(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint.setdefault("id_column", "user_id")
    checkpoint.setdefault("ignored_label_values", [])
    checkpoint.setdefault("ignore_index", -1)
    checkpoint.setdefault("edges_use_node_ids", False)
    checkpoint.setdefault("align_target_features_by_name", False)
    checkpoint.setdefault("headerless_features", False)
    checkpoint.setdefault("separate_evaluation_files", False)
    required = (
        "experiment",
        "source_domain",
        "label_column",
        "sensitive_column",
        "label_mapping",
        "sensitive_mapping",
        "excluded_feature_columns",
        "feature_names",
        "feature_schema",
        "feature_mean",
        "feature_std",
        "model_configuration",
        "encoder_state",
        "bottleneck_state",
        "classifier_state",
    )
    for key in required:
        if key not in checkpoint:
            raise ValueError("Source checkpoint is missing key {}".format(key))

    configuration = checkpoint["model_configuration"]
    if int(configuration["class_num"]) != CLASS_NUM:
        raise ValueError("GCN-SHOT currently requires exactly two classes")
    encoder, bottleneck, classifier = build_models_from_configuration(
        configuration, device
    )
    encoder.load_state_dict(checkpoint["encoder_state"])
    bottleneck.load_state_dict(checkpoint["bottleneck_state"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    return checkpoint, encoder, bottleneck, classifier


def train_target(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint, encoder, bottleneck, classifier = load_source_artifact(
        args.source_checkpoint, device
    )

    if checkpoint["excluded_feature_columns"]:
        print(
            "Loading target graph inputs without {}. No source CSV/edge path "
            "is accepted by this target stage...".format(
                ", ".join(checkpoint["excluded_feature_columns"])
            )
        )
    else:
        print(
            "Loading headerless target model features. Standalone target label "
            "and sensitive files remain unread during adaptation..."
        )
    target_graph = load_graph_inputs(
        args.csv,
        args.edges,
        expected_feature_names=checkpoint["feature_names"],
        excluded_feature_columns=checkpoint["excluded_feature_columns"],
        expected_feature_schema=checkpoint["feature_schema"],
        id_column=(
            checkpoint["id_column"]
            if checkpoint["edges_use_node_ids"]
            else None
        ),
        align_to_expected_schema=checkpoint[
            "align_target_features_by_name"
        ],
        has_header=not checkpoint["headerless_features"],
    )
    feature_mean = checkpoint["feature_mean"].float()
    feature_std = checkpoint["feature_std"].float()
    target_features = standardize_features(
        target_graph.features, feature_mean, feature_std
    ).to(device)
    target_adjacency = target_graph.adjacency.to(device)

    # Cache source-only predictions now, but do not load target labels or the
    # sensitive attribute.
    source_only_predictions, source_only_probabilities = get_predictions(
        encoder,
        bottleneck,
        classifier,
        target_features,
        target_adjacency,
    )
    source_only_counts = torch.bincount(
        source_only_predictions, minlength=CLASS_NUM
    ).tolist()
    print(
        "Source-only target prediction counts (labels still hidden): {}".format(
            source_only_counts
        )
    )

    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False

    optimizer = create_optimizer(
        [
            {"params": encoder.parameters(), "lr": args.lr * 0.1},
            {"params": bottleneck.parameters(), "lr": args.lr},
        ]
    )

    node_count = target_graph.num_nodes
    batch_size = min(args.batch_size, node_count)
    steps_per_epoch = int(math.ceil(float(node_count) / float(batch_size)))
    total_steps = args.epochs * steps_per_epoch
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        pseudo_labels, pseudo_counts, mean_probability = (
            obtain_strict_shot_pseudo_labels(
                encoder,
                bottleneck,
                classifier,
                target_features,
                target_adjacency,
                refine_rounds=args.pseudo_refine_rounds,
                threshold=args.threshold,
                epsilon=args.epsilon,
            )
        )

        encoder.train()
        bottleneck.train()
        classifier.eval()
        node_order = torch.randperm(node_count, device=device)
        epoch_sums = {
            "pseudo_label": 0.0,
            "conditional_entropy": 0.0,
            "marginal_entropy": 0.0,
            "information_maximization": 0.0,
            "total": 0.0,
        }
        seen_nodes = 0

        for start in range(0, node_count, batch_size):
            batch_indices = node_order[start : start + batch_size]
            current_batch_size = int(batch_indices.numel())
            inverse_learning_rate(optimizer, global_step, total_steps)

            _, all_logits = forward_model(
                encoder,
                bottleneck,
                classifier,
                target_features,
                target_adjacency,
            )
            batch_logits = all_logits[batch_indices]
            batch_pseudo_labels = pseudo_labels[batch_indices]
            target_loss, components = strict_shot_loss(
                batch_logits,
                batch_pseudo_labels,
                cls_par=args.cls_par,
                ent_par=args.ent_par,
                epsilon=args.epsilon,
            )

            optimizer.zero_grad()
            target_loss.backward()
            optimizer.step()
            global_step += 1

            for name in epoch_sums:
                epoch_sums[name] += components[name] * current_batch_size
            seen_nodes += current_batch_size

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            averaged = {
                name: value / float(seen_nodes)
                for name, value in epoch_sums.items()
            }
            print(
                "Target epoch {:03d}/{:03d} loss={:.6f} ce={:.6f} "
                "cond_ent={:.6f} marg_ent={:.6f} pseudo_counts={} "
                "mean_prob=[{:.4f}, {:.4f}]".format(
                    epoch,
                    args.epochs,
                    averaged["total"],
                    averaged["pseudo_label"],
                    averaged["conditional_entropy"],
                    averaged["marginal_entropy"],
                    pseudo_counts.tolist(),
                    mean_probability[0].item(),
                    mean_probability[1].item(),
                )
            )

    (
        adapted_representations,
        adapted_predictions,
        adapted_probabilities,
    ) = get_predictions_with_representations(
        encoder,
        bottleneck,
        classifier,
        target_features,
        target_adjacency,
    )

    ensure_directory(args.output_dir)
    adapted_checkpoint_path = os.path.join(
        args.output_dir, "target_shot_model.pt"
    )
    adapted_checkpoint = {
        "format_version": 3,
        "experiment": checkpoint["experiment"],
        "source_checkpoint": os.path.abspath(args.source_checkpoint),
        "source_domain": checkpoint["source_domain"],
        "target_domain": args.target_domain,
        "label_column": checkpoint["label_column"],
        "sensitive_column": checkpoint["sensitive_column"],
        "label_mapping": checkpoint["label_mapping"],
        "ignored_label_values": checkpoint["ignored_label_values"],
        "ignore_index": checkpoint["ignore_index"],
        "sensitive_mapping": checkpoint["sensitive_mapping"],
        "id_column": checkpoint["id_column"],
        "edges_use_node_ids": checkpoint["edges_use_node_ids"],
        "align_target_features_by_name": checkpoint[
            "align_target_features_by_name"
        ],
        "headerless_features": checkpoint["headerless_features"],
        "separate_evaluation_files": checkpoint[
            "separate_evaluation_files"
        ],
        "excluded_feature_columns": list(
            checkpoint["excluded_feature_columns"]
        ),
        "feature_names": list(checkpoint["feature_names"]),
        "feature_schema": list(checkpoint["feature_schema"]),
        "feature_mean": checkpoint["feature_mean"],
        "feature_std": checkpoint["feature_std"],
        "model_configuration": checkpoint["model_configuration"],
        "encoder_state": encoder.state_dict(),
        "bottleneck_state": bottleneck.state_dict(),
        "classifier_state": classifier.state_dict(),
        "shot_configuration": {
            "cls_par": args.cls_par,
            "ent_par": args.ent_par,
            "threshold": args.threshold,
            "pseudo_refine_rounds": args.pseudo_refine_rounds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }
    torch.save(adapted_checkpoint, adapted_checkpoint_path)

    print(
        "\nAdaptation is complete. Loading target {} and {} now, for final "
        "evaluation only...".format(
            checkpoint["label_column"], checkpoint["sensitive_column"]
        )
    )
    if checkpoint["separate_evaluation_files"]:
        if args.label_file is None or args.sensitive_file is None:
            raise ValueError(
                "This source artifact requires target --label_file and "
                "--sensitive_file for final evaluation"
            )
        target_labels, target_sensitive = load_target_evaluation_files(
            args.label_file,
            args.sensitive_file,
            label_name=checkpoint["label_column"],
            sensitive_name=checkpoint["sensitive_column"],
            label_mapping=checkpoint["label_mapping"],
            sensitive_mapping=checkpoint["sensitive_mapping"],
            ignored_label_values=checkpoint["ignored_label_values"],
            ignore_index=checkpoint["ignore_index"],
        )
    else:
        target_labels, target_sensitive = load_target_evaluation_columns(
            args.csv,
            label_column=checkpoint["label_column"],
            sensitive_column=checkpoint["sensitive_column"],
            label_mapping=checkpoint["label_mapping"],
            sensitive_mapping=checkpoint["sensitive_mapping"],
            ignored_label_values=checkpoint["ignored_label_values"],
            ignore_index=checkpoint["ignore_index"],
        )
    if target_labels.numel() != target_graph.num_nodes:
        raise ValueError("Target graph and evaluation columns have different lengths")
    evaluation_mask = target_labels != checkpoint["ignore_index"]
    if torch.sum(evaluation_mask).item() == 0:
        raise ValueError("The target graph has no labeled evaluation nodes")
    evaluation_labels = target_labels[evaluation_mask]
    evaluation_sensitive = target_sensitive[evaluation_mask]
    validate_binary_tensor(
        evaluation_labels,
        "labeled target {}".format(checkpoint["label_column"]),
    )
    print(
        "Target metrics use {} labeled nodes; {} unlabeled nodes remain "
        "excluded from evaluation only.".format(
            int(torch.sum(evaluation_mask).item()),
            int(torch.sum(~evaluation_mask).item()),
        )
    )

    source_only_results = evaluate_predictions(
        evaluation_labels,
        source_only_predictions[evaluation_mask],
        positive_scores=source_only_probabilities[evaluation_mask, 1],
        sensitive=evaluation_sensitive,
        sensitive_name=checkpoint["sensitive_column"],
    )
    adapted_results = evaluate_predictions(
        evaluation_labels,
        adapted_predictions[evaluation_mask],
        positive_scores=adapted_probabilities[evaluation_mask, 1],
        sensitive=evaluation_sensitive,
        sensitive_name=checkpoint["sensitive_column"],
    )
    visualization_export = export_target_visualization_embeddings(
        args.output_dir,
        adapted_representations,
        adapted_predictions,
        target_sensitive,
        evaluation_mask,
    )
    print(
        "Saved target visualization representations to {}".format(
            visualization_export["feat_path"]
        )
    )
    print(
        "Saved target visualization joint labels to {}".format(
            visualization_export["labels_path"]
        )
    )
    results = {
        "experiment": checkpoint["experiment"],
        "seed": args.seed,
        "source_domain": checkpoint["source_domain"],
        "target_domain": args.target_domain,
        "label_column": checkpoint["label_column"],
        "sensitive_attribute": checkpoint["sensitive_column"],
        "label_mapping": checkpoint["label_mapping"],
        "ignored_label_values": checkpoint["ignored_label_values"],
        "ignore_index": checkpoint["ignore_index"],
        "sensitive_mapping": checkpoint["sensitive_mapping"],
        "source_only": source_only_results,
        "strict_shot": adapted_results,
        "reported_metrics": {
            "source_only": headline_metrics(source_only_results),
            "strict_shot": headline_metrics(adapted_results),
        },
        "reported_metric_unit": "percentage points",
        "prediction_counts": {
            "source_only": torch.bincount(
                source_only_predictions, minlength=CLASS_NUM
            ).tolist(),
            "strict_shot": torch.bincount(
                adapted_predictions, minlength=CLASS_NUM
            ).tolist(),
        },
        "evaluated_prediction_counts": {
            "source_only": torch.bincount(
                source_only_predictions[evaluation_mask], minlength=CLASS_NUM
            ).tolist(),
            "strict_shot": torch.bincount(
                adapted_predictions[evaluation_mask], minlength=CLASS_NUM
            ).tolist(),
        },
        "target_node_counts": {
            "total": target_graph.num_nodes,
            "evaluated_labeled": int(torch.sum(evaluation_mask).item()),
            "ignored_unlabeled": int(torch.sum(~evaluation_mask).item()),
        },
        "mean_probabilities": {
            "source_only": source_only_probabilities.mean(dim=0).tolist(),
            "strict_shot": adapted_probabilities.mean(dim=0).tolist(),
        },
        "feature_names": list(checkpoint["feature_names"]),
        "excluded_feature_columns": list(
            checkpoint["excluded_feature_columns"]
        ),
        "target_labels_and_sensitive_loaded_after_adaptation": True,
        "target_evaluation_storage": (
            "separate_files"
            if checkpoint["separate_evaluation_files"]
            else "csv_columns"
        ),
        "visualization_export": visualization_export,
    }
    results["reported_metrics_display"] = {
        "source_only": format_headline_metrics(
            results["reported_metrics"]["source_only"]
        ),
        "strict_shot": format_headline_metrics(
            results["reported_metrics"]["strict_shot"]
        ),
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)

    print("Saved adapted model to {}".format(adapted_checkpoint_path))
    print("Saved final metrics to {}".format(results_path))
    print_result_block("Target source-only results", source_only_results)
    print_result_block("Target strict SHOT results", adapted_results)
    print_headline_metrics(
        "Target source-only headline metrics",
        results["reported_metrics"]["source_only"],
    )
    print_headline_metrics(
        "Target strict SHOT headline metrics",
        results["reported_metrics"]["strict_shot"],
    )


def summarize_repeated_values(values):
    if len(values) < 2:
        raise ValueError("At least two runs are required to compute sample std")
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(
        len(values) - 1
    )
    std = math.sqrt(variance)
    return {
        "mean": mean,
        "std": std,
        "values": list(values),
        "mean_percent": format_percentage_points(mean),
        "std_percent": format_percentage_points(std),
        "display": "{} +/- {}".format(
            format_percentage_points(mean), format_percentage_points(std)
        ),
    }


def aggregate_results(args):
    if len(args.results) != args.expected_runs:
        raise ValueError(
            "Expected {} result files, received {}".format(
                args.expected_runs, len(args.results)
            )
        )

    runs = []
    for result_path in args.results:
        with open(result_path, "r") as stream:
            result = json.load(stream)
        if "seed" not in result:
            raise ValueError("Result file has no seed: {}".format(result_path))
        if "reported_metrics" not in result:
            raise ValueError(
                "Result file has no reported_metrics: {}".format(result_path)
            )
        for key in (
            "experiment",
            "label_column",
            "sensitive_attribute",
            "label_mapping",
            "sensitive_mapping",
        ):
            if key not in result:
                raise ValueError(
                    "Result file is missing {}: {}".format(key, result_path)
                )
        runs.append(
            {
                "seed": int(result["seed"]),
                "path": os.path.abspath(result_path),
                "experiment": result["experiment"],
                "label_column": result["label_column"],
                "sensitive_attribute": result["sensitive_attribute"],
                "label_mapping": result["label_mapping"],
                "ignored_label_values": result.get("ignored_label_values", []),
                "sensitive_mapping": result["sensitive_mapping"],
                "reported_metrics": result["reported_metrics"],
            }
        )

    runs = sorted(runs, key=lambda item: item["seed"])
    actual_seeds = [run["seed"] for run in runs]
    expected_seeds = list(range(1, args.expected_runs + 1))
    if actual_seeds != expected_seeds:
        raise ValueError(
            "Expected seeds {}, found {}".format(expected_seeds, actual_seeds)
        )

    for key in (
        "experiment",
        "label_column",
        "sensitive_attribute",
        "label_mapping",
        "ignored_label_values",
        "sensitive_mapping",
    ):
        values = [run[key] for run in runs]
        if any(value != values[0] for value in values[1:]):
            raise ValueError("Five runs disagree on {}: {}".format(key, values))

    experiment = runs[0]["experiment"]
    label_column = runs[0]["label_column"]
    sensitive_attribute = runs[0]["sensitive_attribute"]
    label_mapping = runs[0]["label_mapping"]
    ignored_label_values = runs[0]["ignored_label_values"]
    sensitive_mapping = runs[0]["sensitive_mapping"]
    positive_labels = [
        raw_value
        for raw_value, encoded_value in label_mapping.items()
        if int(encoded_value) == 1
    ]
    if not positive_labels:
        raise ValueError("Label mapping must define at least one positive value")

    model_names = ("source_only", "strict_shot")
    metric_names = ("accuracy", "AUC_ROC", "dp", "eo")
    aggregated = {}
    for model_name in model_names:
        aggregated[model_name] = {}
        for metric_name in metric_names:
            values = [
                float(run["reported_metrics"][model_name][metric_name])
                for run in runs
            ]
            aggregated[model_name][metric_name] = summarize_repeated_values(
                values
            )

    summary = {
        "experiment": "{}_5seeds".format(experiment),
        "seeds": actual_seeds,
        "number_of_runs": len(runs),
        "label_column": label_column,
        "sensitive_attribute": sensitive_attribute,
        "label_mapping": label_mapping,
        "ignored_label_values": ignored_label_values,
        "sensitive_mapping": sensitive_mapping,
        "standard_deviation": "sample standard deviation (n-1)",
        "metric_unit": "percentage points",
        "metric_definitions": {
            "accuracy": (
                "macro mean of per-target-class accuracies (binary balanced "
                "accuracy); higher is better"
            ),
            "AUC_ROC": (
                "macro one-vs-rest target-class ROC-AUC using P({} in {}) "
                "for the positive class and its complement for the negative "
                "class; higher is better".format(
                    label_column, sorted(positive_labels)
                )
            ),
            "dp": (
                "absolute demographic parity difference across {} groups; "
                "lower is better".format(sensitive_attribute)
            ),
            "eo": (
                "absolute equal opportunity difference (TPR gap) across {} "
                "groups; lower is better".format(sensitive_attribute)
            ),
        },
        "aggregate": aggregated,
        "runs": runs,
    }

    ensure_parent_directory(args.output)
    with open(args.output, "w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    text_output = os.path.splitext(args.output)[0] + ".txt"
    with open(text_output, "w") as stream:
        stream.write("model\tmetric\tmean +/- std\n")
        for model_name in model_names:
            for metric_name in metric_names:
                metric = aggregated[model_name][metric_name]
                stream.write(
                    "{}\t{}\t{}\n".format(
                        model_name,
                        metric_name,
                        metric["display"],
                    )
                )

    print("\nFive-seed aggregate (percentage-point mean +/- sample std)")
    print("model          metric      mean +/- std")
    for model_name in model_names:
        for metric_name in metric_names:
            metric = aggregated[model_name][metric_name]
            print(
                "{:<14} {:<10} {}".format(
                    model_name,
                    metric_name,
                    metric["display"],
                )
            )
    print("Saved aggregate JSON to {}".format(args.output))
    print("Saved aggregate table to {}".format(text_output))


def add_runtime_arguments(parser):
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="PyTorch device, for example cuda:0 or cpu",
    )
    parser.add_argument("--seed", type=int, default=2020)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="GCN-based strict SHOT for binary graph node classification"
    )
    subparsers = parser.add_subparsers(dest="stage")

    source_parser = subparsers.add_parser(
        "source", help="Train a labeled binary source graph model"
    )
    add_runtime_arguments(source_parser)
    source_parser.add_argument("--csv", type=str, required=True)
    source_parser.add_argument("--edges", type=str, required=True)
    source_parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Standalone source label file for headerless feature datasets",
    )
    source_parser.add_argument(
        "--headerless_features",
        action="store_true",
        help="Treat --csv as a headerless all-numeric feature matrix",
    )
    source_parser.add_argument("--checkpoint", type=str, required=True)
    source_parser.add_argument(
        "--experiment",
        type=str,
        default="bailA_2_to_bailA_1_strict_shot_gcn",
    )
    source_parser.add_argument(
        "--source_domain", type=str, default="bailA_2"
    )
    source_parser.add_argument("--id_column", type=str, default="user_id")
    source_parser.add_argument("--label_column", type=str, default="RECID")
    source_parser.add_argument(
        "--sensitive_column", type=str, default="WHITE"
    )
    source_parser.add_argument("--negative_label", type=str, default="0")
    source_parser.add_argument("--positive_label", type=str, default="1")
    source_parser.add_argument(
        "--additional_positive_labels",
        type=str,
        nargs="*",
        default=[],
        help="Additional raw label values collapsed into binary class 1",
    )
    source_parser.add_argument(
        "--ignored_labels",
        type=str,
        nargs="*",
        default=[],
        help="Raw unlabeled values retained in the graph but excluded from metrics",
    )
    source_parser.add_argument("--ignore_index", type=int, default=-1)
    source_parser.add_argument(
        "--sensitive_group_zero", type=str, default="0"
    )
    source_parser.add_argument(
        "--sensitive_group_one", type=str, default="1"
    )
    source_parser.add_argument(
        "--edges_use_node_ids",
        action="store_true",
        help="Map edge endpoints through the CSV identifier column",
    )
    source_parser.add_argument(
        "--align_target_features_by_name",
        action="store_true",
        help=(
            "Reorder target columns to the source schema, zero-fill missing "
            "source columns, and ignore target-only columns"
        ),
    )
    source_parser.add_argument("--epochs", type=int, default=300)
    source_parser.add_argument("--lr", type=float, default=1e-2)
    source_parser.add_argument("--hidden_dim", type=int, default=128)
    source_parser.add_argument("--encoder_dim", type=int, default=128)
    source_parser.add_argument("--bottleneck_dim", type=int, default=256)
    source_parser.add_argument("--dropout", type=float, default=0.5)
    source_parser.add_argument("--smooth", type=float, default=0.1)
    source_parser.add_argument("--train_ratio", type=float, default=0.8)
    source_parser.add_argument(
        "--validation_ratio", type=float, default=0.1
    )
    source_parser.add_argument("--log_interval", type=int, default=10)

    target_parser = subparsers.add_parser(
        "target",
        help="Adapt to an unlabeled target graph and then evaluate",
    )
    add_runtime_arguments(target_parser)
    target_parser.add_argument("--csv", type=str, required=True)
    target_parser.add_argument("--edges", type=str, required=True)
    target_parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Standalone target labels loaded only after adaptation",
    )
    target_parser.add_argument(
        "--sensitive_file",
        type=str,
        default=None,
        help="Standalone target sensitive values loaded only after adaptation",
    )
    target_parser.add_argument(
        "--source_checkpoint", type=str, required=True
    )
    target_parser.add_argument("--output_dir", type=str, required=True)
    target_parser.add_argument(
        "--target_domain", type=str, default="bailA_1"
    )
    target_parser.add_argument("--epochs", type=int, default=15)
    target_parser.add_argument("--batch_size", type=int, default=256)
    target_parser.add_argument("--lr", type=float, default=1e-2)
    target_parser.add_argument("--cls_par", type=float, default=0.3)
    target_parser.add_argument("--ent_par", type=float, default=1.0)
    target_parser.add_argument("--threshold", type=int, default=0)
    target_parser.add_argument(
        "--pseudo_refine_rounds", type=int, default=2
    )
    target_parser.add_argument("--epsilon", type=float, default=1e-5)
    target_parser.add_argument("--log_interval", type=int, default=1)

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate seed 1 through seed 5 target result files",
    )
    aggregate_parser.add_argument(
        "--results", type=str, nargs="+", required=True
    )
    aggregate_parser.add_argument("--output", type=str, required=True)
    aggregate_parser.add_argument("--expected_runs", type=int, default=5)
    return parser


def validate_arguments(args):
    if args.stage is None:
        raise ValueError("Choose either the source or target stage")
    if args.stage == "aggregate":
        if args.expected_runs != 5:
            raise ValueError("This experiment requires exactly five runs")
        if len(args.results) != args.expected_runs:
            raise ValueError(
                "aggregate requires exactly {} result files".format(
                    args.expected_runs
                )
            )
        return
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if args.stage == "source":
        special_columns = (
            args.id_column,
            args.label_column,
            args.sensitive_column,
        )
        if len(set(special_columns)) != 3:
            raise ValueError(
                "id, label, and sensitive columns must be distinct"
            )
        if args.smooth < 0.0 or args.smooth >= 1.0:
            raise ValueError("smooth must be in [0, 1)")
        if args.headerless_features and args.label_file is None:
            raise ValueError(
                "Headerless source features require a standalone --label_file"
            )
        if args.headerless_features and args.edges_use_node_ids:
            raise ValueError(
                "Headerless feature files have no identifier column; use local "
                "row-index edge endpoints"
            )
    if args.stage == "target":
        if (args.label_file is None) != (args.sensitive_file is None):
            raise ValueError(
                "Target --label_file and --sensitive_file must be supplied together"
            )
        if args.batch_size <= 1:
            raise ValueError("batch_size must be greater than 1")
        if args.cls_par <= 0.0:
            raise ValueError(
                "Full SHOT requires cls_par > 0; use 0 only for SHOT-IM ablation"
            )
        if args.ent_par <= 0.0:
            raise ValueError("Strict SHOT requires ent_par > 0")
        if args.pseudo_refine_rounds <= 0:
            raise ValueError("pseudo_refine_rounds must be positive")


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)
    if args.stage == "source":
        train_source(args)
    elif args.stage == "target":
        train_target(args)
    elif args.stage == "aggregate":
        aggregate_results(args)
    else:
        raise ValueError("Unsupported stage {}".format(args.stage))


if __name__ == "__main__":
    main()
