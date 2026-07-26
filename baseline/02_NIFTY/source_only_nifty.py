"""Strict two-stage Source-only NIFTY-GCN for cross-domain evaluation.

This entry point reuses the dataset processing implemented in
``source_only_gcn.py`` so GCN and NIFTY-GCN see exactly the same source/target
features, graphs, label masks, sensitive-attribute exclusions, normalization,
and evaluation metrics.

Unlike the plain-GCN baseline, NIFTY appends the binary sensitive attribute to
its model input so the second augmented view can apply the paper's
counterfactual transformation ``s <- 1 - s``. The shared GCN data-processing
module remains unchanged; this behavior is local to this NIFTY entry point.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj

from source_only_gcn import (
    PROJECT_ROOT,
    SourceMinMaxScaler,
    calculate_target_metrics,
    load_domain,
    resolve_data_dir,
    resolve_device,
    schema_for_domain,
    set_seed,
)
from visualization_export import (
    method_name_for_seed,
    save_target_visualization_embeddings,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "source_only_nifty_gcn_bailA_2_seed1.pt"
)
DEFAULT_VISUALIZATION_EMBEDDINGS_ROOT = (
    PROJECT_ROOT / "visualization" / "embeddings"
)


def _spectral_linear(input_dim: int, output_dim: int) -> nn.Module:
    linear = nn.Linear(input_dim, output_dim)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    return spectral_norm(linear)


class NiftyGCN(nn.Module):
    """Repository-compatible SSF/NIFTY model with a one-layer GCN encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        projection_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)

        self.projector = nn.Sequential(
            _spectral_linear(hidden_dim, projection_hidden_dim),
            nn.BatchNorm1d(projection_hidden_dim),
            nn.ReLU(inplace=True),
            _spectral_linear(projection_hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.predictor = nn.Sequential(
            _spectral_linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            _spectral_linear(hidden_dim, hidden_dim),
        )
        self.classifier = _spectral_linear(hidden_dim, 1)

    def encode(self, features: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(features, edge_index)

    def project(self, embeddings: Tensor) -> Tensor:
        return self.projector(embeddings)

    def predict_projection(self, projections: Tensor) -> Tensor:
        return self.predictor(projections)

    def classify(self, embeddings: Tensor) -> Tensor:
        return self.classifier(embeddings).squeeze(-1)

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        return self.encode(features, edge_index)


def validate_probability(value: float, name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")


def perturb_features(
    features: Tensor,
    probability: float,
    sensitive_index: int,
    flip_sensitive: bool,
) -> Tensor:
    """Perturb non-sensitive dimensions and optionally flip the sensitive one.

    A Bernoulli mask chooses feature dimensions and one Gaussian offset per
    chosen dimension is broadcast across nodes, matching NIFTY's r ⊙ delta
    construction. The sensitive dimension is always excluded from this noise
    mask and is changed only through the exact binary flip ``1 - s``.
    """

    validate_probability(probability, "feature perturbation probability")
    if sensitive_index < 0 or sensitive_index >= features.shape[1]:
        raise IndexError(
            f"Sensitive index {sensitive_index} is outside feature dimension "
            f"{features.shape[1]}."
        )
    perturbed = features.clone()
    if probability > 0.0:
        feature_mask = torch.rand(
            features.shape[1],
            device=features.device,
        ) < probability
        feature_mask[sensitive_index] = False
        if feature_mask.any():
            offsets = torch.randn(
                features.shape[1],
                dtype=features.dtype,
                device=features.device,
            )
            perturbed[:, feature_mask] += offsets[feature_mask]

    if flip_sensitive:
        sensitive_values = perturbed[:, sensitive_index]
        if not torch.all((sensitive_values == 0) | (sensitive_values == 1)):
            raise ValueError(
                "Sensitive feature must be binary 0/1 before counterfactual flip."
            )
        perturbed[:, sensitive_index] = 1.0 - sensitive_values
    return perturbed


def negative_cosine_similarity(prediction: Tensor, target: Tensor) -> Tensor:
    return -F.cosine_similarity(
        prediction,
        target.detach(),
        dim=-1,
    ).mean()


def make_augmented_views(
    features: Tensor,
    edge_index: Tensor,
    drop_edge_rate_1: float,
    drop_edge_rate_2: float,
    drop_feature_rate_1: float,
    drop_feature_rate_2: float,
    sensitive_index: int,
) -> Sequence[Tensor]:
    validate_probability(drop_edge_rate_1, "drop_edge_rate_1")
    validate_probability(drop_edge_rate_2, "drop_edge_rate_2")
    validate_probability(drop_feature_rate_1, "drop_feature_rate_1")
    validate_probability(drop_feature_rate_2, "drop_feature_rate_2")

    edge_index_1 = dropout_adj(
        edge_index,
        p=drop_edge_rate_1,
        training=True,
    )[0]
    edge_index_2 = dropout_adj(
        edge_index,
        p=drop_edge_rate_2,
        training=True,
    )[0]
    features_1 = perturb_features(
        features,
        drop_feature_rate_1,
        sensitive_index=sensitive_index,
        flip_sensitive=False,
    )
    features_2 = perturb_features(
        features,
        drop_feature_rate_2,
        sensitive_index=sensitive_index,
        flip_sensitive=True,
    )
    return features_1, edge_index_1, features_2, edge_index_2


def _checkpoint_payload(
    model: NiftyGCN,
    scaler: SourceMinMaxScaler,
    source,
    args: argparse.Namespace,
    final_similarity_loss: float,
    final_classification_loss: float,
) -> Dict[str, object]:
    return {
        "format_version": 1,
        "experiment": "source_only_nifty_gcn",
        "source_domain": source.domain,
        "feature_columns": source.feature_columns,
        "data_config": {
            "schema_name": source.schema_name,
            "id_column": source.id_column,
            "sensitive_column": source.sensitive_column,
            "label_column": source.label_column,
        },
        "normalization": {
            "type": "source_min_max_to_minus_one_one",
            **scaler.to_dict(),
        },
        "model_config": {
            "base_feature_dim": int(source.features.shape[1]),
            "input_dim": int(source.features.shape[1] + 1),
            "hidden_dim": int(args.hidden),
            "projection_hidden_dim": int(args.proj_hidden),
            "encoder": "single_layer_gcn",
            "uses_sensitive_feature": True,
            "sensitive_feature_position": "last",
            "uses_sensitive_counterfactual_augmentation": True,
            "counterfactual_operation": "s <- 1 - s",
        },
        "training_config": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "similarity_coefficient": float(args.sim_coeff),
            "drop_edge_rate_1": float(args.drop_edge_rate_1),
            "drop_edge_rate_2": float(args.drop_edge_rate_2),
            "drop_feature_rate_1": float(args.drop_feature_rate_1),
            "drop_feature_rate_2": float(args.drop_feature_rate_2),
            "uses_all_source_nodes_for_similarity": True,
            "uses_all_available_source_labels": True,
            "source_graph_nodes": int(source.labels.shape[0]),
            "source_labeled_nodes": int(source.labeled_mask.sum()),
            "has_source_validation_split": False,
        },
        "final_similarity_loss": float(final_similarity_loss),
        "final_classification_loss": float(final_classification_loss),
        "model_state_dict": model.state_dict(),
    }


def train_source(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1.")
    validate_probability(args.sim_coeff, "sim_coeff")
    validate_probability(args.drop_edge_rate_1, "drop_edge_rate_1")
    validate_probability(args.drop_edge_rate_2, "drop_edge_rate_2")
    validate_probability(args.drop_feature_rate_1, "drop_feature_rate_1")
    validate_probability(args.drop_feature_rate_2, "drop_feature_rate_2")

    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / f"source_only_nifty_gcn_{args.source_domain}_seed{args.seed}.pt"
        )

    source_data_dir = resolve_data_dir(args.source_domain, args.data_dir)
    source = load_domain(
        data_dir=source_data_dir,
        domain=args.source_domain,
        include_sensitive=True,
    )
    if source.sensitive is None:
        raise AssertionError(
            f"Source {source.sensitive_column} was not loaded for NIFTY training."
        )
    scaler = SourceMinMaxScaler.fit(source.features)
    normalized_features = scaler.transform(source.features)

    base_features = torch.as_tensor(
        normalized_features,
        dtype=torch.float32,
        device=device,
    )
    sensitive_feature = torch.as_tensor(
        source.sensitive,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    features = torch.cat([base_features, sensitive_feature], dim=1)
    sensitive_index = features.shape[1] - 1
    labels = torch.as_tensor(source.labels, dtype=torch.float32, device=device)
    labeled_mask = torch.as_tensor(
        source.labeled_mask,
        dtype=torch.bool,
        device=device,
    )
    edge_index = source.edge_index.to(device)

    model = NiftyGCN(
        input_dim=features.shape[1],
        hidden_dim=args.hidden,
        projection_hidden_dim=args.proj_hidden,
    ).to(device)

    similarity_parameters = (
        list(model.encoder.parameters())
        + list(model.projector.parameters())
        + list(model.predictor.parameters())
    )
    classification_parameters = (
        list(model.encoder.parameters())
        + list(model.classifier.parameters())
    )
    similarity_optimizer = torch.optim.Adam(
        similarity_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    classification_optimizer = torch.optim.Adam(
        classification_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Stage: train")
    print("Method: Source-only NIFTY-GCN")
    print(f"Source domain: {source.domain}")
    print(f"Source graph nodes: {labels.numel()}")
    print(f"Source labeled nodes used for BCE: {int(labeled_mask.sum().item())}")
    print(f"Non-sensitive input features: {base_features.shape[1]}")
    print(
        f"Sensitive input feature: {source.sensitive_column} "
        f"at index {sensitive_index}"
    )
    print(f"Total NIFTY input features: {features.shape[1]}")
    print("Sensitive counterfactual augmentation: enabled (s <- 1 - s)")
    print(f"Device: {device}")

    final_similarity_loss = float("nan")
    final_classification_loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        features_1, edges_1, features_2, edges_2 = make_augmented_views(
            features=features,
            edge_index=edge_index,
            drop_edge_rate_1=args.drop_edge_rate_1,
            drop_edge_rate_2=args.drop_edge_rate_2,
            drop_feature_rate_1=args.drop_feature_rate_1,
            drop_feature_rate_2=args.drop_feature_rate_2,
            sensitive_index=sensitive_index,
        )

        similarity_optimizer.zero_grad()
        embeddings_1 = model.encode(features_1, edges_1)
        embeddings_2 = model.encode(features_2, edges_2)
        projections_1 = model.project(embeddings_1)
        projections_2 = model.project(embeddings_2)
        predictions_1 = model.predict_projection(projections_1)
        predictions_2 = model.predict_projection(projections_2)
        similarity_loss = args.sim_coeff * (
            negative_cosine_similarity(predictions_1, projections_2) / 2.0
            + negative_cosine_similarity(predictions_2, projections_1) / 2.0
        )
        similarity_loss.backward()
        similarity_optimizer.step()

        # Clear shared-encoder gradients left by the similarity update before
        # the supervised classification update.
        classification_optimizer.zero_grad()
        embeddings_1 = model.encode(features_1, edges_1)
        embeddings_2 = model.encode(features_2, edges_2)
        logits_1 = model.classify(embeddings_1)
        logits_2 = model.classify(embeddings_2)
        classification_loss = (1.0 - args.sim_coeff) * (
            F.binary_cross_entropy_with_logits(
                logits_1[labeled_mask],
                labels[labeled_mask],
            )
            / 2.0
            + F.binary_cross_entropy_with_logits(
                logits_2[labeled_mask],
                labels[labeled_mask],
            )
            / 2.0
        )
        classification_loss.backward()
        classification_optimizer.step()

        final_similarity_loss = float(similarity_loss.detach().cpu().item())
        final_classification_loss = float(
            classification_loss.detach().cpu().item()
        )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            total_loss = final_similarity_loss + final_classification_loss
            print(
                f"Epoch {epoch:04d}/{args.epochs:04d} "
                f"similarity={final_similarity_loss:.6f} "
                f"classification={final_classification_loss:.6f} "
                f"total={total_loss:.6f}"
            )

    checkpoint = _checkpoint_payload(
        model=model,
        scaler=scaler,
        source=source,
        args=args,
        final_similarity_loss=final_similarity_loss,
        final_classification_loss=final_classification_loss,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved trained Source-only NIFTY-GCN: {checkpoint_path.resolve()}")


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("experiment") != "source_only_nifty_gcn":
        raise ValueError(
            f"{checkpoint_path} is not a source_only_nifty_gcn checkpoint."
        )
    if checkpoint.get("format_version") != 1:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint.get('format_version')}."
        )
    return checkpoint


def test_target(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device=device)

    feature_columns = list(checkpoint["feature_columns"])
    target_schema = schema_for_domain(args.target_domain)
    if target_schema.sensitive_column in feature_columns:
        raise ValueError(
            "Invalid checkpoint: sensitive attribute "
            f"{target_schema.sensitive_column} appears in model features."
        )

    target_data_dir = resolve_data_dir(args.target_domain, args.data_dir)
    target = load_domain(
        data_dir=target_data_dir,
        domain=args.target_domain,
        include_sensitive=True,
        expected_feature_columns=feature_columns,
    )
    if target.sensitive is None:
        raise AssertionError(
            f"Target {target.sensitive_column} was not loaded for evaluation."
        )

    data_config = checkpoint["data_config"]
    if data_config.get("schema_name") != target.schema_name:
        raise ValueError(
            "Source and target domains use incompatible schemas: "
            f"source={data_config.get('schema_name')}, "
            f"target={target.schema_name}."
        )
    if data_config.get("label_column") != target.label_column:
        raise ValueError("Source and target label columns do not match.")
    if data_config.get("sensitive_column") != target.sensitive_column:
        raise ValueError("Source and target sensitive columns do not match.")

    normalization = checkpoint["normalization"]
    if normalization.get("type") != "source_min_max_to_minus_one_one":
        raise ValueError(
            f"Unsupported normalization type: {normalization.get('type')}."
        )
    scaler = SourceMinMaxScaler.from_dict(normalization)
    base_target_features = scaler.transform(target.features)
    sensitive_feature = target.sensitive.astype(np.float32, copy=False).reshape(-1, 1)
    target_features = np.concatenate(
        [base_target_features, sensitive_feature],
        axis=1,
    )

    model_config = checkpoint["model_config"]
    if model_config.get("uses_sensitive_feature") is not True:
        raise ValueError(
            "Checkpoint does not certify that NIFTY used the sensitive feature."
        )
    if model_config.get("sensitive_feature_position") != "last":
        raise ValueError("Unsupported sensitive-feature position in checkpoint.")
    if model_config.get("uses_sensitive_counterfactual_augmentation") is not True:
        raise ValueError(
            "Checkpoint does not certify sensitive counterfactual augmentation."
        )
    if int(model_config["base_feature_dim"]) != base_target_features.shape[1]:
        raise ValueError(
            "Target non-sensitive feature dimension does not match source NIFTY."
        )
    if int(model_config["input_dim"]) != target_features.shape[1]:
        raise ValueError(
            "Target feature dimension does not match the trained source model."
        )

    model = NiftyGCN(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        projection_hidden_dim=int(model_config["projection_hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = torch.as_tensor(
        target_features,
        dtype=torch.float32,
        device=device,
    )
    edge_index = target.edge_index.to(device)
    with torch.inference_mode():
        embeddings = model.encode(features, edge_index)
        logits = model.classify(embeddings).detach().cpu().numpy()
        representations = embeddings.detach().cpu().numpy()

    evaluation_mask = target.labeled_mask
    metrics = calculate_target_metrics(
        labels=target.labels[evaluation_mask],
        logits=logits[evaluation_mask],
        sensitive=target.sensitive[evaluation_mask],
        sensitive_column=target.sensitive_column,
        label_column=target.label_column,
    )
    training_config = checkpoint.get("training_config", {})
    if "seed" not in training_config:
        raise ValueError("Checkpoint does not record its training seed.")
    seed = int(training_config["seed"])
    visualization_method = method_name_for_seed("NIFTY-GCN", seed)
    visualization_predictions = (logits[evaluation_mask] > 0.0).astype(np.int64)
    feat_path, labels_path = save_target_visualization_embeddings(
        embeddings_root=args.visualization_embeddings_root,
        method=visualization_method,
        dataset=target.schema_name,
        representations=representations[evaluation_mask],
        predictions=visualization_predictions,
        sensitive=target.sensitive[evaluation_mask],
    )

    print("Stage: test")
    print("Method: Source-only NIFTY-GCN")
    print(f"Trained source domain: {checkpoint['source_domain']}")
    print(f"Target domain: {target.domain}")
    print(f"Sensitive model input: {target.sensitive_column} (original target values)")
    print("Training counterfactual operation: s <- 1 - s")
    print(f"Target graph nodes: {target.labels.shape[0]}")
    print(f"Target labeled nodes evaluated: {int(evaluation_mask.sum())}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC_AUC: {metrics['roc_auc']:.4f}")
    print(f"Parity: {metrics['parity']:.4f}")
    print(f"Equality: {metrics['equality']:.4f}")
    print(f"Saved visualization representations: {feat_path.resolve()}")
    print(f"Saved visualization labels: {labels_path.resolve()}")

    if args.output_json is not None:
        result = {
            "experiment": "source_only_nifty_gcn",
            "source_domain": checkpoint["source_domain"],
            "target_domain": target.domain,
            "dataset_schema": target.schema_name,
            "label_column": target.label_column,
            "sensitive_column": target.sensitive_column,
            "uses_sensitive_feature": True,
            "uses_sensitive_counterfactual_augmentation": True,
            "checkpoint": str(args.checkpoint.resolve()),
            "target_graph_nodes": int(target.labels.shape[0]),
            "target_labeled_nodes": int(evaluation_mask.sum()),
            "target_nodes": int(evaluation_mask.sum()),
            "metrics": metrics,
            "visualization": {
                "method": visualization_method,
                "dataset": target.schema_name,
                "seed": seed,
                "feat_path": str(feat_path.resolve()),
                "labels_path": str(labels_path.resolve()),
                "representations_shape": [
                    int(evaluation_mask.sum()),
                    int(representations.shape[1]),
                ],
                "label_encoding": {
                    "0": "predicted_Y=1,S=0",
                    "1": "predicted_Y=1,S=1",
                    "2": "predicted_Y=0,S=0",
                    "3": "predicted_Y=0,S=1",
                },
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Saved target metrics: {args.output_json.resolve()}")


def _append_training_arguments(command: List[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--hidden",
            str(args.hidden),
            "--proj-hidden",
            str(args.proj_hidden),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--sim-coeff",
            str(args.sim_coeff),
            "--drop-edge-rate-1",
            str(args.drop_edge_rate_1),
            "--drop-edge-rate-2",
            str(args.drop_edge_rate_2),
            "--drop-feature-rate-1",
            str(args.drop_feature_rate_1),
            "--drop-feature-rate-2",
            str(args.drop_feature_rate_2),
            "--device",
            args.device,
            "--log-every",
            str(args.log_every),
        ]
    )


def run_source_to_target(args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve()
    source_data_dir = resolve_data_dir(args.source_domain, args.source_data_dir)
    target_data_dir = resolve_data_dir(args.target_domain, args.target_data_dir)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / f"source_only_nifty_gcn_{args.source_domain}_seed{args.seed}.pt"
        )

    train_command = [
        sys.executable,
        "-u",
        str(script_path),
        "train",
        "--source-domain",
        args.source_domain,
        "--data-dir",
        str(source_data_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--seed",
        str(args.seed),
    ]
    _append_training_arguments(train_command, args)

    test_command = [
        sys.executable,
        "-u",
        str(script_path),
        "test",
        "--target-domain",
        args.target_domain,
        "--data-dir",
        str(target_data_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--device",
        args.device,
        "--visualization-embeddings-root",
        str(args.visualization_embeddings_root),
    ]
    if args.output_json is not None:
        test_command.extend(["--output-json", str(args.output_json)])

    print("Combined run: launching isolated NIFTY source training.", flush=True)
    subprocess.run(train_command, check=True)
    print(
        "Source training exited; launching a fresh frozen target test.",
        flush=True,
    )
    subprocess.run(test_command, check=True)


def run_multiple_seeds(args: argparse.Namespace) -> None:
    seeds = list(args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicate values: {seeds}")
    if args.std_ddof >= len(seeds):
        raise ValueError("--std-ddof must be smaller than the number of seeds.")

    source_data_dir = resolve_data_dir(args.source_domain, args.source_data_dir)
    target_data_dir = resolve_data_dir(args.target_domain, args.target_data_dir)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    metric_names = ["accuracy", "roc_auc", "parity", "equality"]
    per_seed_results = []
    for run_index, seed in enumerate(seeds, start=1):
        checkpoint_path = (
            args.checkpoint_dir
            / f"source_only_nifty_gcn_{args.source_domain}_seed{seed}.pt"
        )
        result_path = (
            args.results_dir
            / (
                f"source_only_nifty_gcn_{args.source_domain}_to_"
                f"{args.target_domain}_seed{seed}.json"
            )
        )
        command = [
            sys.executable,
            "-u",
            str(script_path),
            "run",
            "--source-domain",
            args.source_domain,
            "--target-domain",
            args.target_domain,
            "--source-data-dir",
            str(source_data_dir),
            "--target-data-dir",
            str(target_data_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output-json",
            str(result_path),
            "--seed",
            str(seed),
            "--visualization-embeddings-root",
            str(args.visualization_embeddings_root),
        ]
        _append_training_arguments(command, args)

        print(f"\n===== NIFTY seed {seed} ({run_index}/{len(seeds)}) =====", flush=True)
        subprocess.run(command, check=True)
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        metrics = result.get("metrics", {})
        missing = [name for name in metric_names if name not in metrics]
        if missing:
            raise ValueError(f"{result_path} is missing metrics: {missing}.")
        per_seed_results.append(
            {
                "seed": int(seed),
                "checkpoint": str(checkpoint_path.resolve()),
                "result_json": str(result_path.resolve()),
                "visualization": result["visualization"],
                "metrics": {
                    name: float(metrics[name])
                    for name in metric_names
                },
            }
        )

    aggregate = {}
    for metric_name in metric_names:
        values = np.asarray(
            [item["metrics"][metric_name] for item in per_seed_results],
            dtype=np.float64,
        )
        aggregate[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=args.std_ddof)),
            "values": values.tolist(),
        }

    if args.summary_json is None:
        seed_label = "-".join(str(seed) for seed in seeds)
        summary_path = (
            args.results_dir
            / (
                f"source_only_nifty_gcn_{args.source_domain}_to_"
                f"{args.target_domain}_seeds_{seed_label}_summary.json"
            )
        )
    else:
        summary_path = args.summary_json

    summary = {
        "experiment": "source_only_nifty_gcn_multi_seed",
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "seeds": [int(seed) for seed in seeds],
        "number_of_runs": len(seeds),
        "standard_deviation_ddof": int(args.std_ddof),
        "uses_sensitive_feature": True,
        "uses_sensitive_counterfactual_augmentation": True,
        "counterfactual_operation": "s <- 1 - s",
        "per_seed": per_seed_results,
        "aggregate": aggregate,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n===== Multi-seed NIFTY target summary =====")
    print(f"Seeds: {seeds}")
    print(f"Standard deviation: numpy.std(ddof={args.std_ddof})")
    print(
        f"Accuracy: {100.0 * aggregate['accuracy']['mean']:.2f} "
        f"± {100.0 * aggregate['accuracy']['std']:.2f}"
    )
    print(
        f"ROC_AUC: {100.0 * aggregate['roc_auc']['mean']:.2f} "
        f"± {100.0 * aggregate['roc_auc']['std']:.2f}"
    )
    print(
        f"Parity: {100.0 * aggregate['parity']['mean']:.2f} "
        f"± {100.0 * aggregate['parity']['std']:.2f}"
    )
    print(
        f"Equality: {100.0 * aggregate['equality']['mean']:.2f} "
        f"± {100.0 * aggregate['equality']['std']:.2f}"
    )
    print(f"Saved multi-seed summary: {summary_path.resolve()}")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--proj-hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sim-coeff", type=float, default=0.6)
    parser.add_argument("--drop-edge-rate-1", type=float, default=0.001)
    parser.add_argument("--drop-edge-rate-2", type=float, default=0.001)
    parser.add_argument("--drop-feature-rate-1", type=float, default=0.1)
    parser.add_argument("--drop-feature-rate-2", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict Source-only NIFTY-GCN with isolated source training and "
            "frozen target testing."
        )
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--source-domain", default="bailA_2")
    train_parser.add_argument("--data-dir", type=Path, default=None)
    train_parser.add_argument("--checkpoint", type=Path, default=None)
    train_parser.add_argument("--seed", type=int, default=1)
    add_training_arguments(train_parser)
    train_parser.set_defaults(handler=train_source)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--target-domain", default="bailA_1")
    test_parser.add_argument("--data-dir", type=Path, default=None)
    test_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    test_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    test_parser.add_argument("--output-json", type=Path, default=None)
    test_parser.add_argument(
        "--visualization-embeddings-root",
        type=Path,
        default=DEFAULT_VISUALIZATION_EMBEDDINGS_ROOT,
        help=(
            "Root for per-seed <method>/<dataset>/feat.npz and labels.npz "
            "exports."
        ),
    )
    test_parser.set_defaults(handler=test_target)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--source-domain", default="bailA_2")
    run_parser.add_argument("--target-domain", default="bailA_1")
    run_parser.add_argument("--source-data-dir", type=Path, default=None)
    run_parser.add_argument("--target-data-dir", type=Path, default=None)
    run_parser.add_argument("--checkpoint", type=Path, default=None)
    run_parser.add_argument("--output-json", type=Path, default=None)
    run_parser.add_argument(
        "--visualization-embeddings-root",
        type=Path,
        default=DEFAULT_VISUALIZATION_EMBEDDINGS_ROOT,
    )
    run_parser.add_argument("--seed", type=int, default=1)
    add_training_arguments(run_parser)
    run_parser.set_defaults(handler=run_source_to_target)

    multi_parser = subparsers.add_parser("run-seeds")
    multi_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
    )
    multi_parser.add_argument("--source-domain", default="bailA_2")
    multi_parser.add_argument("--target-domain", default="bailA_1")
    multi_parser.add_argument("--source-data-dir", type=Path, default=None)
    multi_parser.add_argument("--target-data-dir", type=Path, default=None)
    multi_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    multi_parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    multi_parser.add_argument("--summary-json", type=Path, default=None)
    multi_parser.add_argument(
        "--visualization-embeddings-root",
        type=Path,
        default=DEFAULT_VISUALIZATION_EMBEDDINGS_ROOT,
    )
    multi_parser.add_argument(
        "--std-ddof",
        type=int,
        choices=[0, 1],
        default=0,
    )
    add_training_arguments(multi_parser)
    multi_parser.set_defaults(handler=run_multiple_seeds)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
