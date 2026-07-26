"""GCN-based strict original NRC for BailA, GermanA, Pokec, and Syn.

The experiment is deliberately split into four commands:

``source``
    Reads source model features/edges and labels, trains the source GCN, and
    writes a source artifact.
``target``
    Accepts no source data path. It reads only target model features/edges,
    performs NRC adaptation without labels or sensitive attributes, and writes
    predictions.
``evaluate``
    Runs only after adaptation and is the sole stage that reads target labels
    and sensitive attributes.
``summarize``
    Aggregates the five independent seed result files.

The NRC target objective follows the released implementation: dynamic feature
and score banks, reciprocal-neighbor affinity, expanded neighborhoods,
implicit self-regularization by retaining ego occurrences in the expanded
neighborhood, and prediction-diversity regularization.  The GCN encoder,
bottleneck, and classifier are all updated during target adaptation.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as functional

from baila_data import (
    compute_feature_statistics,
    get_dataset_configuration,
    load_graph_inputs,
    load_label_column,
    load_target_evaluation_columns,
    load_target_evaluation_files,
    load_value_file,
    standardize_features,
)
from gcn import build_gcn_nrc_models, forward_model


CLASS_NUM = 2
METRIC_NAMES = ("accuracy", "roc_auc", "parity", "equality")


def experiment_name(dataset_configuration):
    return "{}_to_{}_strict_original_nrc_gcn".format(
        dataset_configuration["source_domain"],
        dataset_configuration["target_domain"],
    )


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device {} was requested, but CUDA is unavailable".format(
                device_name
            )
        )
    return torch.device(device_name)


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_parent_directory(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        ensure_directory(parent)


def save_json(payload, path):
    ensure_parent_directory(path)
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def save_visualization_embeddings(output_dir, representations, labels):
    """Save the visualization contract used by the AAAI-2026 visualizer."""

    if representations.dim() != 2:
        raise ValueError(
            "Target representations must be 2D, found shape {}".format(
                tuple(representations.shape)
            )
        )
    if labels.dim() != 1:
        raise ValueError(
            "Visualization labels must be 1D, found shape {}".format(
                tuple(labels.shape)
            )
        )
    if representations.size(0) != labels.numel():
        raise ValueError(
            "Visualization representations and labels differ in length: "
            "{} vs {}".format(representations.size(0), labels.numel())
        )
    if not bool(torch.isfinite(representations).all().item()):
        raise ValueError(
            "Target representations contain NaN or Inf; refusing to export"
        )
    if not bool(((labels >= 0) & (labels <= 3)).all().item()):
        raise ValueError(
            "Visualization labels must contain only values 0, 1, 2, or 3"
        )

    ensure_directory(output_dir)
    feature_path = os.path.join(output_dir, "feat.npz")
    label_path = os.path.join(output_dir, "labels.npz")
    representation_array = representations.detach().cpu().numpy()
    label_array = labels.detach().cpu().long().numpy()
    np.savez_compressed(
        feature_path,
        representations=representation_array,
    )
    np.savez_compressed(label_path, labels=label_array)

    # Re-open the files through NumPy so the on-disk contract is checked, not
    # merely the tensors that were passed to the writer.
    with np.load(feature_path, allow_pickle=False) as feature_data:
        saved_representations = np.asarray(feature_data["representations"])
    with np.load(label_path, allow_pickle=False) as label_data:
        saved_labels = np.asarray(label_data["labels"])
    if saved_representations.ndim != 2:
        raise ValueError("Saved representations are not a 2D array")
    if saved_labels.ndim != 1:
        raise ValueError("Saved labels are not a 1D array")
    if saved_representations.shape[0] != saved_labels.shape[0]:
        raise ValueError(
            "Saved representations and labels differ in length: {} vs {}".format(
                saved_representations.shape[0], saved_labels.shape[0]
            )
        )
    if not np.isfinite(saved_representations).all():
        raise ValueError("Saved representations contain NaN or Inf")
    if not np.isin(saved_labels, np.asarray([0, 1, 2, 3])).all():
        raise ValueError("Saved labels contain values outside 0, 1, 2, 3")
    return feature_path, label_path


def state_dict_to_cpu(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


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


def rank_based_roc_auc(labels, positive_scores):
    """Binary ROC-AUC with average ranks for tied prediction scores."""

    labels = labels.detach().cpu().long().view(-1)
    scores = positive_scores.detach().cpu().double().view(-1)
    if labels.numel() != scores.numel():
        raise ValueError("ROC-AUC labels and scores have different lengths")

    positive_count = int(torch.sum(labels == 1).item())
    negative_count = int(torch.sum(labels == 0).item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty(scores.numel(), dtype=torch.double)
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while (
            end < sorted_scores.numel()
            and sorted_scores[end].item() == sorted_scores[start].item()
        ):
            end += 1
        average_rank = (float(start + 1) + float(end)) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = torch.sum(ranks[labels == 1]).item()
    auc = (
        positive_rank_sum
        - float(positive_count * (positive_count + 1)) / 2.0
    ) / float(positive_count * negative_count)
    return float(auc)


def classification_metrics(labels, predictions, probabilities):
    labels = labels.detach().cpu().long().view(-1)
    predictions = predictions.detach().cpu().long().view(-1)
    probabilities = probabilities.detach().cpu().float()
    if labels.numel() != predictions.numel():
        raise ValueError("Labels and predictions have different lengths")
    if probabilities.dim() != 2 or probabilities.size(1) != CLASS_NUM:
        raise ValueError("Expected a [num_nodes, 2] probability tensor")
    if probabilities.size(0) != labels.numel():
        raise ValueError("Labels and probabilities have different lengths")

    return {
        "accuracy": torch.mean((labels == predictions).float()).item(),
        "roc_auc": rank_based_roc_auc(labels, probabilities[:, 1]),
    }


def fairness_metrics(labels, predictions, sensitive):
    labels = labels.detach().cpu().long().view(-1)
    predictions = predictions.detach().cpu().long().view(-1)
    sensitive = sensitive.detach().cpu().long().view(-1)
    if (
        labels.numel() != predictions.numel()
        or labels.numel() != sensitive.numel()
    ):
        raise ValueError("Evaluation tensors have different lengths")
    validate_binary_tensor(sensitive, "target sensitive attribute")

    positive_rates = {}
    true_positive_rates = {}
    for group_index in (0, 1):
        group_mask = sensitive == group_index
        positive_label_mask = group_mask & (labels == 1)
        if int(torch.sum(group_mask).item()) == 0:
            raise ValueError("Sensitive group {} is empty".format(group_index))
        if int(torch.sum(positive_label_mask).item()) == 0:
            raise ValueError(
                "Sensitive group {} has no positive-label samples".format(
                    group_index
                )
            )
        positive_rates[str(group_index)] = torch.mean(
            (predictions[group_mask] == 1).float()
        ).item()
        true_positive_rates[str(group_index)] = torch.mean(
            (predictions[positive_label_mask] == 1).float()
        ).item()

    return {
        "parity": abs(positive_rates["0"] - positive_rates["1"]),
        "equality": abs(
            true_positive_rates["0"] - true_positive_rates["1"]
        ),
        "positive_prediction_rate": positive_rates,
        "true_positive_rate": true_positive_rates,
    }


def evaluate_binary_predictions(labels, predictions, probabilities, sensitive):
    metrics = classification_metrics(labels, predictions, probabilities)
    metrics.update(fairness_metrics(labels, predictions, sensitive))
    return metrics


def build_models_from_configuration(configuration, device):
    encoder, bottleneck, classifier = build_gcn_nrc_models(
        input_dim=int(configuration["input_dim"]),
        hidden_dim=int(configuration["hidden_dim"]),
        encoder_dim=int(configuration["encoder_dim"]),
        bottleneck_dim=int(configuration["bottleneck_dim"]),
        class_num=int(configuration["class_num"]),
        dropout=float(configuration["dropout"]),
    )
    return encoder.to(device), bottleneck.to(device), classifier.to(device)


def get_model_outputs(encoder, bottleneck, classifier, features, adjacency):
    encoder.eval()
    bottleneck.eval()
    classifier.eval()
    with torch.no_grad():
        bottleneck_features, logits = forward_model(
            encoder, bottleneck, classifier, features, adjacency
        )
        probabilities = functional.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    return (
        bottleneck_features.detach(),
        predictions.detach().cpu(),
        probabilities.detach().cpu(),
    )


def train_source(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    dataset_configuration = get_dataset_configuration(args.dataset)
    current_experiment = experiment_name(dataset_configuration)
    label_column = dataset_configuration["label_column"]
    excluded_columns = dataset_configuration["excluded_feature_columns"]

    if dataset_configuration["feature_file_has_header"]:
        print(
            "Loading {} model features with excluded columns {}...".format(
                dataset_configuration["source_domain"],
                list(excluded_columns),
            )
        )
    else:
        print(
            "Loading {} headerless numeric model features; labels and "
            "sensitive values are stored separately...".format(
                dataset_configuration["source_domain"]
            )
        )
    source_graph = load_graph_inputs(
        args.csv,
        args.edges,
        excluded_feature_columns=excluded_columns,
        node_id_column=dataset_configuration["node_id_column"],
        feature_file_has_header=dataset_configuration[
            "feature_file_has_header"
        ],
    )
    print(
        "Loading {} separately for supervised source training...".format(
            label_column
        )
    )
    source_label_storage = dataset_configuration["source_label_storage"]
    if source_label_storage == "csv_column":
        source_labels = load_label_column(
            args.csv,
            label_column,
            value_mapping=dataset_configuration["label_mapping"],
        )
    elif source_label_storage == "separate_file":
        source_labels = load_value_file(
            args.label_file,
            value_name=label_column,
            value_mapping=dataset_configuration["label_mapping"],
        )
    else:
        raise ValueError(
            "Unsupported source label storage {}".format(
                source_label_storage
            )
        )
    if source_graph.num_nodes != source_labels.numel():
        raise ValueError("Source graph and source labels have different lengths")
    unlabeled_label_value = dataset_configuration["unlabeled_label_value"]
    if unlabeled_label_value is None:
        source_labeled_mask = torch.ones_like(source_labels, dtype=torch.bool)
    else:
        source_labeled_mask = source_labels != unlabeled_label_value
    if int(torch.sum(source_labeled_mask).item()) == 0:
        raise ValueError("Source domain has no labeled nodes")
    validate_binary_tensor(
        source_labels[source_labeled_mask],
        "labeled source {}".format(label_column),
    )

    train_mask, validation_mask, test_mask = stratified_masks(
        source_labels,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    normalization = dataset_configuration["normalization"]
    if normalization == "source_train_standardize":
        feature_mean, feature_std = compute_feature_statistics(
            source_graph.features, train_mask
        )
    elif normalization == "none":
        feature_mean = torch.zeros_like(source_graph.features[0])
        feature_std = torch.ones_like(source_graph.features[0])
    elif normalization == "source_full_minmax_minus_one_one":
        feature_min = source_graph.features.min(dim=0)[0]
        feature_max = source_graph.features.max(dim=0)[0]
        feature_mean = (feature_max + feature_min) / 2.0
        half_range = (feature_max - feature_min) / 2.0
        feature_std = torch.where(
            half_range == 0.0,
            torch.ones_like(half_range),
            half_range,
        )
    else:
        raise ValueError(
            "Unsupported normalization mode {}".format(normalization)
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
    best_validation_loss = float("inf")
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
            _, predictions, probabilities = get_model_outputs(
                encoder, bottleneck, classifier, features, adjacency
            )
            validation_metrics = classification_metrics(
                source_labels[validation_mask],
                predictions[validation_mask],
                probabilities[validation_mask],
            )
            encoder.eval()
            bottleneck.eval()
            classifier.eval()
            with torch.no_grad():
                _, validation_logits = forward_model(
                    encoder, bottleneck, classifier, features, adjacency
                )
                validation_loss = functional.cross_entropy(
                    validation_logits[validation_mask.to(device)],
                    labels[validation_mask.to(device)],
                ).item()
            print(
                "Source epoch {:04d}/{:04d} loss={:.6f} val_acc={:.4f} "
                "val_auc={:.4f}".format(
                    epoch,
                    args.epochs,
                    source_loss.item(),
                    validation_metrics["accuracy"],
                    validation_metrics["roc_auc"],
                )
            )
            is_better = validation_metrics["accuracy"] > best_validation_accuracy
            is_tie_but_lower_loss = (
                validation_metrics["accuracy"] == best_validation_accuracy
                and validation_loss < best_validation_loss
            )
            if is_better or is_tie_but_lower_loss:
                best_validation_accuracy = validation_metrics["accuracy"]
                best_validation_loss = validation_loss
                best_states = {
                    "encoder": state_dict_to_cpu(encoder),
                    "bottleneck": state_dict_to_cpu(bottleneck),
                    "classifier": state_dict_to_cpu(classifier),
                    "epoch": epoch,
                }

    if best_states is None:
        raise RuntimeError("Source training did not produce a checkpoint")

    encoder.load_state_dict(best_states["encoder"])
    bottleneck.load_state_dict(best_states["bottleneck"])
    classifier.load_state_dict(best_states["classifier"])
    _, source_predictions, source_probabilities = get_model_outputs(
        encoder, bottleneck, classifier, features, adjacency
    )
    validation_metrics = classification_metrics(
        source_labels[validation_mask],
        source_predictions[validation_mask],
        source_probabilities[validation_mask],
    )
    source_test_metrics = classification_metrics(
        source_labels[test_mask],
        source_predictions[test_mask],
        source_probabilities[test_mask],
    )

    checkpoint = {
        "format_version": 1,
        "experiment": current_experiment,
        "dataset_family": dataset_configuration["dataset_family"],
        "source_domain": dataset_configuration["source_domain"],
        "target_domain": dataset_configuration["target_domain"],
        "label_column": label_column,
        "sensitive_column": dataset_configuration["sensitive_column"],
        "label_mapping": dict(dataset_configuration["label_mapping"]),
        "sensitive_mapping": dict(dataset_configuration["sensitive_mapping"]),
        "normalization": normalization,
        "node_id_column": dataset_configuration["node_id_column"],
        "target_feature_alignment": dataset_configuration[
            "target_feature_alignment"
        ],
        "unlabeled_label_value": unlabeled_label_value,
        "feature_file_has_header": dataset_configuration[
            "feature_file_has_header"
        ],
        "source_label_storage": source_label_storage,
        "target_evaluation_storage": dataset_configuration[
            "target_evaluation_storage"
        ],
        "excluded_feature_columns": list(excluded_columns),
        "feature_names": list(source_graph.feature_names),
        "feature_mean": feature_mean.detach().cpu(),
        "feature_std": feature_std.detach().cpu(),
        "model_configuration": configuration,
        "encoder_state": state_dict_to_cpu(encoder),
        "bottleneck_state": state_dict_to_cpu(bottleneck),
        "classifier_state": state_dict_to_cpu(classifier),
        "seed": args.seed,
        "best_source_epoch": int(best_states["epoch"]),
        "source_validation_metrics": validation_metrics,
        "source_test_metrics": source_test_metrics,
        "source_node_count": source_graph.num_nodes,
        "source_labeled_node_count": int(
            torch.sum(source_labeled_mask).item()
        ),
    }
    ensure_parent_directory(args.checkpoint)
    torch.save(checkpoint, args.checkpoint)

    print("Saved source artifact to {}".format(args.checkpoint))
    print("Model feature columns: {}".format(source_graph.feature_names))
    print("Best source epoch: {}".format(best_states["epoch"]))
    print("Source validation: {}".format(validation_metrics))
    print("Source held-out test: {}".format(source_test_metrics))


def load_source_artifact(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    required = (
        "dataset_family",
        "label_column",
        "sensitive_column",
        "label_mapping",
        "sensitive_mapping",
        "normalization",
        "node_id_column",
        "target_feature_alignment",
        "unlabeled_label_value",
        "feature_file_has_header",
        "source_label_storage",
        "target_evaluation_storage",
        "excluded_feature_columns",
        "feature_names",
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
        raise ValueError("This NRC graph experiment requires exactly two classes")
    encoder, bottleneck, classifier = build_models_from_configuration(
        configuration, device
    )
    encoder.load_state_dict(checkpoint["encoder_state"])
    bottleneck.load_state_dict(checkpoint["bottleneck_state"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    return checkpoint, encoder, bottleneck, classifier


def initialize_memory_banks(
    encoder,
    bottleneck,
    classifier,
    features,
    adjacency,
    maximum_size=0,
):
    bottleneck_features, _, probabilities = get_model_outputs(
        encoder, bottleneck, classifier, features, adjacency
    )
    all_feature_bank = functional.normalize(
        bottleneck_features, p=2, dim=1
    ).detach().clone()
    all_score_bank = probabilities.to(features.device).detach().clone()
    node_count = int(all_feature_bank.size(0))
    all_node_ids = torch.arange(node_count, device=features.device)
    if maximum_size > 0 and maximum_size < node_count:
        selected = torch.randperm(node_count, device=features.device)[
            :maximum_size
        ]
        return (
            all_feature_bank[selected].to(features.device),
            all_score_bank[selected],
            all_node_ids[selected],
            True,
        )
    return (
        all_feature_bank.to(features.device),
        all_score_bank,
        all_node_ids,
        False,
    )


def retrieve_nrc_neighborhoods(
    batch_features,
    batch_indices,
    feature_bank,
    bank_node_ids,
    neighbor_count,
    expanded_neighbor_count,
    nonreciprocal_affinity,
    query_chunk_size,
):
    """Retrieve KNN, reciprocal affinity, and neighbors-of-neighbors."""

    query_chunk_size = max(1, int(query_chunk_size))
    nearest_chunks = []
    expanded_chunks = []
    affinity_chunks = []
    for start in range(0, batch_features.size(0), query_chunk_size):
        end = min(start + query_chunk_size, batch_features.size(0))
        chunk_features = batch_features[start:end]
        chunk_node_ids = batch_indices[start:end]

        similarities = torch.mm(chunk_features, feature_bank.t())
        ego_mask = (
            bank_node_ids.view(1, -1)
            == chunk_node_ids.view(-1, 1)
        )
        similarities.masked_fill_(ego_mask, float("-inf"))
        nearest_positions = torch.topk(
            similarities, k=neighbor_count, dim=1, largest=True
        ).indices

        nearest_features = feature_bank[nearest_positions]
        nearest_node_ids = bank_node_ids[nearest_positions]
        expanded_similarities = torch.matmul(
            nearest_features, feature_bank.t()
        )
        # Exclude each neighbor j from N_M(j), but deliberately retain the
        # query/ego node i if i occurs there. This preserves the official NRC
        # implementation's implicit self-regularization.
        neighbor_self_mask = (
            bank_node_ids.view(1, 1, -1)
            == nearest_node_ids.unsqueeze(-1)
        )
        expanded_similarities.masked_fill_(
            neighbor_self_mask, float("-inf")
        )
        expanded_positions = torch.topk(
            expanded_similarities,
            k=expanded_neighbor_count,
            dim=2,
            largest=True,
        ).indices

        expanded_node_ids = bank_node_ids[expanded_positions]
        ego_node_ids = chunk_node_ids.view(-1, 1, 1)
        reciprocal_matches = (
            expanded_node_ids == ego_node_ids
        ).sum(dim=2).float()
        neighbor_affinity = torch.where(
            reciprocal_matches > 0.0,
            torch.ones_like(reciprocal_matches),
            torch.full_like(
                reciprocal_matches, nonreciprocal_affinity
            ),
        )
        nearest_chunks.append(nearest_positions)
        expanded_chunks.append(expanded_positions)
        affinity_chunks.append(neighbor_affinity)

    return (
        torch.cat(nearest_chunks, dim=0),
        torch.cat(expanded_chunks, dim=0),
        torch.cat(affinity_chunks, dim=0),
    )


def original_nrc_loss(
    probabilities,
    nearest_scores,
    expanded_scores,
    neighbor_affinity,
    expanded_affinity,
    epsilon,
):
    """Paper NRC objective with official-code implicit self regularization."""

    nearest_similarity = torch.sum(
        probabilities.unsqueeze(1) * nearest_scores, dim=2
    )
    neighbor_loss = -torch.mean(
        torch.sum(nearest_similarity * neighbor_affinity, dim=1)
    )

    expanded_similarity = torch.sum(
        probabilities.unsqueeze(1).unsqueeze(1) * expanded_scores,
        dim=3,
    )
    expanded_loss = -torch.mean(
        torch.sum(
            torch.sum(expanded_similarity * expanded_affinity, dim=2),
            dim=1,
        )
    )

    mean_probability = probabilities.mean(dim=0)
    diversity_loss = torch.sum(
        mean_probability * torch.log(mean_probability + epsilon)
    )
    total_loss = neighbor_loss + expanded_loss + diversity_loss
    return total_loss, neighbor_loss, expanded_loss, diversity_loss


def train_target(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    dataset_configuration = get_dataset_configuration(args.dataset)
    current_experiment = experiment_name(dataset_configuration)
    checkpoint, encoder, bottleneck, classifier = load_source_artifact(
        args.source_checkpoint, device
    )
    if checkpoint["dataset_family"] != dataset_configuration["dataset_family"]:
        raise ValueError(
            "Source checkpoint dataset {} does not match requested dataset {}".format(
                checkpoint["dataset_family"],
                dataset_configuration["dataset_family"],
            )
        )
    if int(checkpoint.get("seed", args.seed)) != args.seed:
        raise ValueError(
            "Source checkpoint seed {} does not match target seed {}".format(
                checkpoint.get("seed"), args.seed
            )
        )

    if checkpoint["feature_file_has_header"]:
        print(
            "Loading {} model features with excluded columns {}. "
            "This target command accepts no source CSV or edge path.".format(
                dataset_configuration["target_domain"],
                checkpoint["excluded_feature_columns"],
            )
        )
    else:
        print(
            "Loading {} headerless model features. The standalone target "
            "label and sensitive files are not accepted by this adaptation "
            "command.".format(dataset_configuration["target_domain"])
        )
    target_graph = load_graph_inputs(
        args.csv,
        args.edges,
        expected_feature_names=checkpoint["feature_names"],
        excluded_feature_columns=checkpoint["excluded_feature_columns"],
        feature_alignment=checkpoint["target_feature_alignment"],
        node_id_column=checkpoint["node_id_column"],
        feature_file_has_header=checkpoint["feature_file_has_header"],
    )
    target_features = standardize_features(
        target_graph.features,
        checkpoint["feature_mean"].float(),
        checkpoint["feature_std"].float(),
    ).to(device)
    target_adjacency = target_graph.adjacency.to(device)
    node_count = target_graph.num_nodes
    if args.K >= node_count or args.KK >= node_count:
        raise ValueError("K and KK must both be smaller than target node count")

    _, source_only_predictions, source_only_probabilities = get_model_outputs(
        encoder,
        bottleneck,
        classifier,
        target_features,
        target_adjacency,
    )
    print(
        "Source-only prediction counts while target labels remain hidden: {}".format(
            torch.bincount(
                source_only_predictions, minlength=CLASS_NUM
            ).tolist()
        )
    )

    (
        feature_bank,
        score_bank,
        bank_node_ids,
        bounded_memory_bank,
    ) = initialize_memory_banks(
        encoder,
        bottleneck,
        classifier,
        target_features,
        target_adjacency,
        maximum_size=args.memory_bank_size,
    )
    if feature_bank.size(0) <= max(args.K, args.KK):
        raise ValueError(
            "Memory bank must contain more entries than K and KK"
        )
    print(
        "NRC memory bank: size={} mode={}".format(
            int(feature_bank.size(0)),
            (
                "bounded_fifo_approximation"
                if bounded_memory_bank
                else "full_indexed"
            ),
        )
    )
    optimizer = create_optimizer(
        [
            {"params": encoder.parameters(), "lr": args.lr * 0.1},
            {"params": bottleneck.parameters(), "lr": args.lr},
            {"params": classifier.parameters(), "lr": args.lr},
        ]
    )

    batch_size = min(args.batch_size, node_count)
    steps_per_epoch = int(math.ceil(float(node_count) / float(batch_size)))
    total_steps = args.epochs * steps_per_epoch
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        bottleneck.train()
        classifier.train()
        node_order = torch.randperm(node_count, device=device)
        accumulated = {
            "total": 0.0,
            "neighbor": 0.0,
            "expanded": 0.0,
            "diversity": 0.0,
            "reciprocal": 0.0,
        }
        seen_nodes = 0

        for start in range(0, node_count, batch_size):
            batch_indices = node_order[start : start + batch_size]
            current_batch_size = int(batch_indices.numel())
            inverse_learning_rate(optimizer, global_step, total_steps)

            all_bottleneck_features, all_logits = forward_model(
                encoder,
                bottleneck,
                classifier,
                target_features,
                target_adjacency,
            )
            batch_bottleneck_features = all_bottleneck_features[batch_indices]
            batch_probabilities = functional.softmax(
                all_logits[batch_indices], dim=1
            )

            with torch.no_grad():
                normalized_batch_features = functional.normalize(
                    batch_bottleneck_features, p=2, dim=1
                )
                if bounded_memory_bank:
                    bank_capacity = int(args.memory_bank_size)
                    feature_bank = torch.cat(
                        (feature_bank, normalized_batch_features.detach()),
                        dim=0,
                    )[-bank_capacity:]
                    score_bank = torch.cat(
                        (score_bank, batch_probabilities.detach()), dim=0
                    )[-bank_capacity:]
                    bank_node_ids = torch.cat(
                        (bank_node_ids, batch_indices.detach()), dim=0
                    )[-bank_capacity:]
                else:
                    feature_bank[batch_indices] = (
                        normalized_batch_features.detach()
                    )
                    score_bank[batch_indices] = batch_probabilities.detach()
                (
                    nearest_indices,
                    expanded_indices,
                    neighbor_affinity,
                ) = retrieve_nrc_neighborhoods(
                    normalized_batch_features.detach(),
                    batch_indices,
                    feature_bank,
                    bank_node_ids,
                    neighbor_count=args.K,
                    expanded_neighbor_count=args.KK,
                    nonreciprocal_affinity=args.r,
                    query_chunk_size=args.neighbor_query_chunk_size,
                )
                nearest_scores = score_bank[nearest_indices].detach()
                expanded_scores = score_bank[expanded_indices].detach()
                expanded_affinity = torch.full(
                    expanded_indices.shape,
                    args.r,
                    dtype=batch_probabilities.dtype,
                    device=device,
                )

            (
                target_loss,
                neighbor_loss,
                expanded_loss,
                diversity_loss,
            ) = original_nrc_loss(
                batch_probabilities,
                nearest_scores,
                expanded_scores,
                neighbor_affinity,
                expanded_affinity,
                epsilon=args.epsilon,
            )

            optimizer.zero_grad()
            target_loss.backward()
            optimizer.step()
            global_step += 1

            reciprocal_rate = torch.mean(
                (neighbor_affinity == 1.0).float()
            ).item()
            accumulated["total"] += target_loss.item() * current_batch_size
            accumulated["neighbor"] += neighbor_loss.item() * current_batch_size
            accumulated["expanded"] += expanded_loss.item() * current_batch_size
            accumulated["diversity"] += diversity_loss.item() * current_batch_size
            accumulated["reciprocal"] += reciprocal_rate * current_batch_size
            seen_nodes += current_batch_size

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            averaged = {
                name: value / float(seen_nodes)
                for name, value in accumulated.items()
            }
            print(
                "Target epoch {:03d}/{:03d} total={:.6f} neighbor={:.6f} "
                "expanded={:.6f} diversity={:.6f} reciprocal={:.4f}".format(
                    epoch,
                    args.epochs,
                    averaged["total"],
                    averaged["neighbor"],
                    averaged["expanded"],
                    averaged["diversity"],
                    averaged["reciprocal"],
                )
            )

    (
        adapted_representations,
        adapted_predictions,
        adapted_probabilities,
    ) = get_model_outputs(
        encoder,
        bottleneck,
        classifier,
        target_features,
        target_adjacency,
    )
    adapted_representations = adapted_representations.detach().cpu()

    ensure_directory(args.output_dir)
    target_checkpoint_path = os.path.join(
        args.output_dir, "target_nrc_model.pt"
    )
    evaluation_payload_path = os.path.join(
        args.output_dir, "evaluation_payload.pt"
    )
    target_checkpoint = {
        "format_version": 1,
        "experiment": current_experiment,
        "dataset_family": dataset_configuration["dataset_family"],
        "source_domain": dataset_configuration["source_domain"],
        "target_domain": dataset_configuration["target_domain"],
        "label_column": checkpoint["label_column"],
        "sensitive_column": checkpoint["sensitive_column"],
        "label_mapping": checkpoint["label_mapping"],
        "sensitive_mapping": checkpoint["sensitive_mapping"],
        "normalization": checkpoint["normalization"],
        "node_id_column": checkpoint["node_id_column"],
        "target_feature_alignment": checkpoint["target_feature_alignment"],
        "unlabeled_label_value": checkpoint["unlabeled_label_value"],
        "feature_file_has_header": checkpoint["feature_file_has_header"],
        "source_label_storage": checkpoint["source_label_storage"],
        "target_evaluation_storage": checkpoint[
            "target_evaluation_storage"
        ],
        "source_checkpoint": os.path.abspath(args.source_checkpoint),
        "excluded_feature_columns": list(checkpoint["excluded_feature_columns"]),
        "feature_names": list(checkpoint["feature_names"]),
        "feature_mean": checkpoint["feature_mean"],
        "feature_std": checkpoint["feature_std"],
        "model_configuration": checkpoint["model_configuration"],
        "encoder_state": state_dict_to_cpu(encoder),
        "bottleneck_state": state_dict_to_cpu(bottleneck),
        "classifier_state": state_dict_to_cpu(classifier),
        "nrc_configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "K": args.K,
            "KK": args.KK,
            "r": args.r,
            "epsilon": args.epsilon,
            "memory_bank_size": int(feature_bank.size(0)),
            "memory_bank_mode": (
                "bounded_fifo_approximation"
                if bounded_memory_bank
                else "full_indexed"
            ),
            "neighbor_query_chunk_size": args.neighbor_query_chunk_size,
            "implicit_self_regularization": True,
            "classifier_is_adapted": True,
        },
        "seed": args.seed,
    }
    torch.save(target_checkpoint, target_checkpoint_path)

    evaluation_payload = {
        "format_version": 1,
        "experiment": current_experiment,
        "dataset_family": dataset_configuration["dataset_family"],
        "seed": args.seed,
        "node_count": node_count,
        "source_only_predictions": source_only_predictions,
        "source_only_probabilities": source_only_probabilities,
        "adapted_representations": adapted_representations,
        "nrc_predictions": adapted_predictions,
        "nrc_probabilities": adapted_probabilities,
        "target_label_loaded": False,
        "target_sensitive_attribute_loaded": False,
    }
    torch.save(evaluation_payload, evaluation_payload_path)
    save_json(
        {
            "experiment": target_checkpoint["experiment"],
            "seed": args.seed,
            "target_checkpoint": target_checkpoint_path,
            "evaluation_payload": evaluation_payload_path,
            "target_labels_or_sensitive_attributes_used_for_adaptation": False,
            "source_data_paths_accepted_by_target_stage": False,
            "source_only_prediction_counts": torch.bincount(
                source_only_predictions, minlength=CLASS_NUM
            ).tolist(),
            "nrc_prediction_counts": torch.bincount(
                adapted_predictions, minlength=CLASS_NUM
            ).tolist(),
        },
        os.path.join(args.output_dir, "adaptation_manifest.json"),
    )
    print("Saved fixed-final-epoch NRC model to {}".format(target_checkpoint_path))
    print("Saved label-free evaluation payload to {}".format(evaluation_payload_path))
    print("Target labels and sensitive attributes were not loaded by this process.")


def evaluate_target(args):
    dataset_configuration = get_dataset_configuration(args.dataset)
    current_experiment = experiment_name(dataset_configuration)
    print(
        "Adaptation is already complete; loading target {} and {} now...".format(
            dataset_configuration["label_column"],
            dataset_configuration["sensitive_column"],
        )
    )
    payload = torch.load(args.prediction_payload, map_location="cpu")
    required = (
        "dataset_family",
        "seed",
        "node_count",
        "source_only_predictions",
        "source_only_probabilities",
        "adapted_representations",
        "nrc_predictions",
        "nrc_probabilities",
    )
    for key in required:
        if key not in payload:
            raise ValueError("Prediction payload is missing key {}".format(key))
    if payload["dataset_family"] != dataset_configuration["dataset_family"]:
        raise ValueError(
            "Prediction payload dataset {} does not match requested dataset {}".format(
                payload["dataset_family"],
                dataset_configuration["dataset_family"],
            )
        )

    evaluation_storage = dataset_configuration[
        "target_evaluation_storage"
    ]
    if evaluation_storage == "csv_columns":
        labels, sensitive = load_target_evaluation_columns(
            args.csv,
            label_column=dataset_configuration["label_column"],
            sensitive_column=dataset_configuration["sensitive_column"],
            label_mapping=dataset_configuration["label_mapping"],
            sensitive_mapping=dataset_configuration["sensitive_mapping"],
        )
    elif evaluation_storage == "separate_files":
        labels, sensitive = load_target_evaluation_files(
            args.label_file,
            args.sensitive_file,
            label_name=dataset_configuration["label_column"],
            sensitive_name=dataset_configuration["sensitive_column"],
            label_mapping=dataset_configuration["label_mapping"],
            sensitive_mapping=dataset_configuration["sensitive_mapping"],
        )
    else:
        raise ValueError(
            "Unsupported target evaluation storage {}".format(
                evaluation_storage
            )
        )
    unlabeled_label_value = dataset_configuration["unlabeled_label_value"]
    if unlabeled_label_value is None:
        evaluation_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        evaluation_mask = labels != unlabeled_label_value
    # Negative sensitive values are reserved for unknown groups if a future
    # dataset configuration uses them. Such nodes remain in the graph and NRC
    # adaptation but are excluded from fairness/classification metrics.
    evaluation_mask = evaluation_mask & (sensitive >= 0)
    if int(torch.sum(evaluation_mask).item()) == 0:
        raise ValueError("Target domain has no eligible evaluation nodes")
    validate_binary_tensor(
        labels[evaluation_mask],
        "target {}".format(dataset_configuration["label_column"]),
    )
    validate_binary_tensor(
        sensitive[evaluation_mask],
        "target {}".format(dataset_configuration["sensitive_column"]),
    )
    if labels.numel() != int(payload["node_count"]):
        raise ValueError("Target labels and prediction payload differ in length")
    adapted_representations = payload["adapted_representations"]
    if adapted_representations.dim() != 2:
        raise ValueError(
            "Adapted target representations must be 2D, found {}".format(
                tuple(adapted_representations.shape)
            )
        )
    if adapted_representations.size(0) != labels.numel():
        raise ValueError(
            "Adapted representations and target labels differ in length"
        )

    source_only_metrics = evaluate_binary_predictions(
        labels[evaluation_mask],
        payload["source_only_predictions"][evaluation_mask],
        payload["source_only_probabilities"][evaluation_mask],
        sensitive[evaluation_mask],
    )
    nrc_metrics = evaluate_binary_predictions(
        labels[evaluation_mask],
        payload["nrc_predictions"][evaluation_mask],
        payload["nrc_probabilities"][evaluation_mask],
        sensitive[evaluation_mask],
    )
    visualization_labels = torch.full(
        (int(torch.sum(evaluation_mask).item()),),
        -1,
        dtype=torch.long,
    )
    valid_labels = labels[evaluation_mask]
    valid_sensitive = sensitive[evaluation_mask]
    visualization_labels[(valid_labels == 1) & (valid_sensitive == 0)] = 0
    visualization_labels[(valid_labels == 1) & (valid_sensitive == 1)] = 1
    visualization_labels[(valid_labels == 0) & (valid_sensitive == 0)] = 2
    visualization_labels[(valid_labels == 0) & (valid_sensitive == 1)] = 3
    export_directory = args.export_dir
    if export_directory is None:
        export_directory = os.path.dirname(os.path.abspath(args.output))
    visualization_feature_path, visualization_label_path = (
        save_visualization_embeddings(
            export_directory,
            adapted_representations[evaluation_mask],
            visualization_labels,
        )
    )
    results = {
        "experiment": current_experiment,
        "dataset_family": dataset_configuration["dataset_family"],
        "seed": int(payload["seed"]),
        "source_only": source_only_metrics,
        "nrc": nrc_metrics,
        "target_node_count": int(labels.numel()),
        "evaluated_node_count": int(torch.sum(evaluation_mask).item()),
        "excluded_from_metrics_count": int(
            labels.numel() - torch.sum(evaluation_mask).item()
        ),
        "visualization_feature_path": os.path.abspath(
            visualization_feature_path
        ),
        "visualization_label_path": os.path.abspath(
            visualization_label_path
        ),
        "visualization_node_count": int(visualization_labels.numel()),
        "visualization_feature_dim": int(
            adapted_representations.size(1)
        ),
        "target_label_and_sensitive_attribute_loaded_only_in_final_evaluation": True,
    }
    save_json(results, args.output)
    print("Saved final result to {}".format(args.output))
    print("Saved visualization features to {}".format(visualization_feature_path))
    print("Saved visualization labels to {}".format(visualization_label_path))
    print(json.dumps(results, indent=2, sort_keys=True))


def summarize_results(args):
    if len(args.results) != 5:
        raise ValueError(
            "Exactly five result files are required, found {}".format(
                len(args.results)
            )
        )
    records = []
    for result_path in args.results:
        with open(result_path, "r") as stream:
            record = json.load(stream)
        if "seed" not in record or "source_only" not in record or "nrc" not in record:
            raise ValueError("Malformed result file: {}".format(result_path))
        records.append(record)

    seeds = [int(record["seed"]) for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds in result files: {}".format(seeds))
    if sorted(seeds) != [1, 2, 3, 4, 5]:
        raise ValueError("Expected seeds 1,2,3,4,5, found {}".format(seeds))
    experiments = [record.get("experiment") for record in records]
    if len(set(experiments)) != 1 or experiments[0] is None:
        raise ValueError(
            "Result files belong to different experiments: {}".format(
                experiments
            )
        )

    summary = {
        "experiment": experiments[0],
        "dataset_family": records[0].get("dataset_family"),
        "seeds": seeds,
        "run_count": len(records),
        "variance_definition": {
            "variance": "population variance: sum((x-mean)^2)/n",
            "plus_minus": "population standard deviation: sqrt(variance)",
        },
        "source_only": {},
        "nrc": {},
    }

    for method_name in ("source_only", "nrc"):
        for metric_name in METRIC_NAMES:
            values = [
                float(record[method_name][metric_name]) for record in records
            ]
            mean_value = sum(values) / float(len(values))
            squared_deviations = [
                (value - mean_value) ** 2 for value in values
            ]
            population_variance = sum(squared_deviations) / float(len(values))
            population_standard_deviation = math.sqrt(population_variance)
            display = "{:.2f}% +/- {:.2f}%".format(
                mean_value * 100.0,
                population_standard_deviation * 100.0,
            )
            summary[method_name][metric_name] = {
                "values": values,
                "mean": mean_value,
                "variance": population_variance,
                "display": display,
            }

    save_json(summary, args.output)
    print("Saved five-seed summary to {}".format(args.output))
    for method_name in ("source_only", "nrc"):
        print("\n{}".format(method_name.upper()))
        for metric_name in METRIC_NAMES:
            metric = summary[method_name][metric_name]
            print("  {:>9s}: {}".format(metric_name, metric["display"]))


def add_runtime_arguments(parser):
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="PyTorch device, for example cuda:0 or cpu",
    )
    parser.add_argument("--seed", type=int, default=1)


def add_dataset_argument(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("bailA", "germanA", "pokec", "syn"),
        default="bailA",
        help="Dataset family and preprocessing protocol",
    )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "GCN-based strict original NRC for BailA, GermanA, Pokec, or Syn"
        )
    )
    subparsers = parser.add_subparsers(dest="stage")

    source_parser = subparsers.add_parser(
        "source", help="Train the configured labeled source-domain GCN"
    )
    add_runtime_arguments(source_parser)
    add_dataset_argument(source_parser)
    source_parser.add_argument("--csv", type=str, required=True)
    source_parser.add_argument("--edges", type=str, required=True)
    source_parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Standalone source label file for datasets such as Syn",
    )
    source_parser.add_argument("--checkpoint", type=str, required=True)
    source_parser.add_argument("--epochs", type=int, default=300)
    source_parser.add_argument("--lr", type=float, default=1e-2)
    source_parser.add_argument("--hidden_dim", type=int, default=128)
    source_parser.add_argument("--encoder_dim", type=int, default=128)
    source_parser.add_argument("--bottleneck_dim", type=int, default=256)
    source_parser.add_argument("--dropout", type=float, default=0.5)
    source_parser.add_argument("--smooth", type=float, default=0.1)
    source_parser.add_argument("--train_ratio", type=float, default=0.8)
    source_parser.add_argument("--validation_ratio", type=float, default=0.1)
    source_parser.add_argument("--log_interval", type=int, default=10)

    target_parser = subparsers.add_parser(
        "target", help="Adapt to the configured unlabeled target using NRC"
    )
    add_runtime_arguments(target_parser)
    add_dataset_argument(target_parser)
    target_parser.add_argument("--csv", type=str, required=True)
    target_parser.add_argument("--edges", type=str, required=True)
    target_parser.add_argument("--source_checkpoint", type=str, required=True)
    target_parser.add_argument("--output_dir", type=str, required=True)
    target_parser.add_argument("--epochs", type=int, default=15)
    target_parser.add_argument("--batch_size", type=int, default=64)
    target_parser.add_argument("--lr", type=float, default=1e-3)
    target_parser.add_argument("--K", type=int, default=5)
    target_parser.add_argument("--KK", type=int, default=5)
    target_parser.add_argument("--r", type=float, default=0.1)
    target_parser.add_argument("--epsilon", type=float, default=1e-5)
    target_parser.add_argument(
        "--memory_bank_size",
        type=int,
        default=0,
        help=(
            "0 uses NRC's exact all-target indexed bank; positive values "
            "use a bounded FIFO approximation"
        ),
    )
    target_parser.add_argument(
        "--neighbor_query_chunk_size",
        type=int,
        default=64,
        help="Query chunk size used only to limit KNN peak memory",
    )
    target_parser.add_argument("--log_interval", type=int, default=1)

    evaluation_parser = subparsers.add_parser(
        "evaluate",
        help="Load target labels/sensitive attributes only after adaptation",
    )
    add_dataset_argument(evaluation_parser)
    evaluation_parser.add_argument("--csv", type=str, default=None)
    evaluation_parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Standalone target label file used only in final evaluation",
    )
    evaluation_parser.add_argument(
        "--sensitive_file",
        type=str,
        default=None,
        help="Standalone target sensitive file used only in final evaluation",
    )
    evaluation_parser.add_argument(
        "--prediction_payload", type=str, required=True
    )
    evaluation_parser.add_argument("--output", type=str, required=True)
    evaluation_parser.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help=(
            "Directory for feat.npz and labels.npz; defaults to the folder "
            "containing --output"
        ),
    )

    summary_parser = subparsers.add_parser(
        "summarize", help="Aggregate seed result JSON files"
    )
    summary_parser.add_argument("--results", nargs="+", required=True)
    summary_parser.add_argument("--output", type=str, required=True)
    return parser


def validate_arguments(args):
    if args.stage is None:
        raise ValueError("Choose source, target, evaluate, or summarize")
    if args.stage in ("source", "target"):
        if args.epochs <= 0:
            raise ValueError("epochs must be positive")
        if args.log_interval <= 0:
            raise ValueError("log_interval must be positive")
    if args.stage == "source":
        if args.smooth < 0.0 or args.smooth >= 1.0:
            raise ValueError("smooth must be in [0, 1)")
        dataset_configuration = get_dataset_configuration(args.dataset)
        source_label_storage = dataset_configuration[
            "source_label_storage"
        ]
        if source_label_storage == "separate_file" and args.label_file is None:
            raise ValueError(
                "Dataset {} requires --label_file for source training".format(
                    args.dataset
                )
            )
        if source_label_storage == "csv_column" and args.label_file is not None:
            raise ValueError(
                "Dataset {} stores source labels in --csv; do not pass "
                "--label_file".format(args.dataset)
            )
    if args.stage == "target":
        if args.batch_size <= 1:
            raise ValueError("batch_size must be greater than 1")
        if args.K <= 0 or args.KK <= 0:
            raise ValueError("K and KK must be positive")
        if args.r < 0.0:
            raise ValueError("r must be non-negative")
        if args.memory_bank_size < 0:
            raise ValueError("memory_bank_size must be non-negative")
        if args.neighbor_query_chunk_size <= 0:
            raise ValueError("neighbor_query_chunk_size must be positive")
    if args.stage == "evaluate":
        dataset_configuration = get_dataset_configuration(args.dataset)
        evaluation_storage = dataset_configuration[
            "target_evaluation_storage"
        ]
        if evaluation_storage == "csv_columns":
            if args.csv is None:
                raise ValueError(
                    "Dataset {} requires --csv for final evaluation".format(
                        args.dataset
                    )
                )
            if args.label_file is not None or args.sensitive_file is not None:
                raise ValueError(
                    "Dataset {} stores evaluation values in --csv; do not "
                    "pass standalone evaluation files".format(args.dataset)
                )
        elif evaluation_storage == "separate_files":
            if args.label_file is None or args.sensitive_file is None:
                raise ValueError(
                    "Dataset {} requires --label_file and --sensitive_file "
                    "for final evaluation".format(args.dataset)
                )
        else:
            raise ValueError(
                "Unsupported target evaluation storage {}".format(
                    evaluation_storage
                )
            )


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_arguments(args)
    if args.stage == "source":
        train_source(args)
    elif args.stage == "target":
        train_target(args)
    elif args.stage == "evaluate":
        evaluate_target(args)
    elif args.stage == "summarize":
        summarize_results(args)
    else:
        raise ValueError("Unsupported stage {}".format(args.stage))


if __name__ == "__main__":
    main()
