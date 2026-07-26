"""Strict two-stage Source-only GCN for Bail domain transfer.

The two stages are intentionally separate processes:

1. ``train`` reads only the labeled source graph and saves a model artifact.
2. ``test`` reads only that artifact and the target graph. It never loads the
   source CSV/edge files, creates no optimizer, and performs no parameter
   updates.
3. ``run-seeds`` repeats the isolated train/test workflow for seeds 1--5 and
   reports the mean and standard deviation of all target metrics.

Node identifiers and dataset-specific sensitive attributes are excluded from
model features. The sensitive attribute is loaded only by the test stage and
is used only for the final statistical-parity and equal-opportunity metrics.
"""

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, to_undirected

from visualization_export import (
    method_name_for_seed,
    save_target_visualization_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "source_only_gcn_bailA_2_seed1.pt"
)
DEFAULT_VISUALIZATION_EMBEDDINGS_ROOT = (
    PROJECT_ROOT / "visualization" / "embeddings"
)


@dataclass(frozen=True)
class DatasetSchema:
    """Dataset-specific columns and binary-value conversions."""

    name: str
    directory_name: str
    id_column: str
    sensitive_column: str
    label_column: str
    extra_excluded_columns: Tuple[str, ...]
    label_mapping: Dict[object, int]
    sensitive_mapping: Dict[object, int]
    unlabeled_label_value: Optional[int]


BAIL_SCHEMA = DatasetSchema(
    name="bailA",
    directory_name="bailA",
    id_column="user_id",
    sensitive_column="WHITE",
    label_column="RECID",
    extra_excluded_columns=(),
    label_mapping={0: 0, 1: 1},
    sensitive_mapping={0: 0, 1: 1},
    unlabeled_label_value=None,
)

GERMAN_SCHEMA = DatasetSchema(
    name="germanA",
    directory_name="germanA",
    id_column="user_id",
    sensitive_column="Gender",
    label_column="GoodCustomer",
    # These are also removed by the repository's original German loader.
    extra_excluded_columns=("PurposeOfLoan", "OtherLoansAtStore"),
    label_mapping={-1: 0, 1: 1},
    sensitive_mapping={"Male": 0, "Female": 1},
    unlabeled_label_value=None,
)

SYN_SCHEMA = DatasetSchema(
    name="syn",
    directory_name="syn",
    # Synthetic nodes are identified by their row position; labels and the
    # sensitive attribute live in separate text files rather than CSV columns.
    id_column="row_index",
    sensitive_column="sens",
    label_column="label",
    extra_excluded_columns=(),
    label_mapping={0: 0, 1: 1},
    sensitive_mapping={0: 0, 1: 1},
    unlabeled_label_value=None,
)

POKEC_SCHEMA = DatasetSchema(
    name="pokec",
    # Pokec is special: each domain has its own directory, so
    # resolve_data_dir uses the full domain name instead of this value.
    directory_name="pokec",
    id_column="user_id",
    sensitive_column="region",
    label_column="I_am_working_in_field",
    # pokec_z and pokec_n use slightly different one-hot vocabularies. Drop
    # every domain-specific column so that both domains share 264 features.
    extra_excluded_columns=(
        "zberatelstvo",
        "hackovanie",
        "vtacik",
        "plave",
        "niekto",
        "slobodny",
        "alternativne",
        "alternativa",
        "horolezectvo",
        "bezkovanie",
        "surfing",
        "literaturu o umeni a architekture",
        "madarsky",
    ),
    # -1 denotes an unlabeled node. Values greater than 0 are the positive
    # class, matching the repository's original Pokec preprocessing.
    label_mapping={-1: -1, 0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
    sensitive_mapping={0: 0, 1: 1},
    unlabeled_label_value=-1,
)


def schema_for_domain(domain: str) -> DatasetSchema:
    if domain == "bailA" or domain.startswith("bailA_"):
        return BAIL_SCHEMA
    if domain == "germanA" or domain.startswith("germanA_"):
        return GERMAN_SCHEMA
    if domain.startswith("syn-"):
        return SYN_SCHEMA
    if domain in {"pokec_z", "pokec_n"}:
        return POKEC_SCHEMA
    raise ValueError(
        f"Unsupported domain {domain!r}. Supported families are bailA_* and "
        "germanA_*, plus syn-* and Pokec domains pokec_z/pokec_n."
    )


def resolve_data_dir(domain: str, configured: Optional[Path]) -> Path:
    if configured is not None:
        return configured
    schema = schema_for_domain(domain)
    if schema.name == "pokec":
        return PROJECT_ROOT / "dataset" / domain
    return PROJECT_ROOT / "dataset" / schema.directory_name


@dataclass
class DomainData:
    """One graph domain after parsing, before feature normalization."""

    domain: str
    features: np.ndarray
    labels: np.ndarray
    edge_index: Tensor
    feature_columns: List[str]
    schema_name: str
    id_column: str
    sensitive_column: str
    label_column: str
    labeled_mask: np.ndarray
    sensitive: Optional[np.ndarray] = None


class SourceMinMaxScaler:
    """Min-max scaler fitted on source features and reused on the target."""

    def __init__(self, minimum: np.ndarray, maximum: np.ndarray) -> None:
        self.minimum = np.asarray(minimum, dtype=np.float32)
        self.maximum = np.asarray(maximum, dtype=np.float32)
        if self.minimum.shape != self.maximum.shape:
            raise ValueError("Scaler minimum and maximum have different shapes.")

    @classmethod
    def fit(cls, features: np.ndarray) -> "SourceMinMaxScaler":
        return cls(
            minimum=features.min(axis=0),
            maximum=features.max(axis=0),
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] != self.minimum.shape[0]:
            raise ValueError(
                "Feature dimension does not match the fitted source scaler: "
                f"got {features.shape[1]}, expected {self.minimum.shape[0]}."
            )

        span = self.maximum - self.minimum
        # A constant source feature is mapped to zero in both domains instead
        # of producing a division-by-zero NaN.
        constant_mask = span == 0
        safe_span = span.copy()
        safe_span[constant_mask] = 1.0

        transformed = 2.0 * (features - self.minimum) / safe_span - 1.0
        transformed[:, constant_mask] = 0.0
        return transformed.astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Sequence[float]]) -> "SourceMinMaxScaler":
        return cls(
            minimum=np.asarray(state["minimum"], dtype=np.float32),
            maximum=np.asarray(state["maximum"], dtype=np.float32),
        )


class SourceOnlyGCN(nn.Module):
    """The repository's vanilla GCN: one GCNConv plus one linear classifier."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, features: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(features, edge_index)

    def classify(self, embeddings: Tensor) -> Tensor:
        return self.classifier(embeddings).squeeze(-1)

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        embeddings = self.encode(features, edge_index)
        return self.classify(embeddings)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These flags improve reproducibility. Some PyG scatter operations can
    # still be nondeterministic on particular CUDA/PyTorch combinations.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(requested)


def _validate_binary(values: np.ndarray, column: str, domain: str) -> None:
    unique_values = set(np.unique(values).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"{domain}: {column} must be binary 0/1, got {sorted(unique_values)}."
        )


def _encode_binary_column(
    frame: pd.DataFrame,
    column: str,
    mapping: Dict[object, int],
    domain: str,
) -> np.ndarray:
    encoded = frame[column].map(mapping)
    if encoded.isna().any():
        unsupported_values = sorted(
            {str(value) for value in frame.loc[encoded.isna(), column].unique()}
        )
        raise ValueError(
            f"{domain}: unsupported values in {column}: {unsupported_values}."
        )
    values = encoded.to_numpy(dtype=np.int64)
    _validate_binary(values, column, domain)
    return values


def _encode_label_column(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    domain: str,
) -> Tuple[np.ndarray, np.ndarray]:
    encoded = frame[schema.label_column].map(schema.label_mapping)
    if encoded.isna().any():
        unsupported_values = sorted(
            {
                str(value)
                for value in frame.loc[encoded.isna(), schema.label_column].unique()
            }
        )
        raise ValueError(
            f"{domain}: unsupported values in {schema.label_column}: "
            f"{unsupported_values}."
        )

    labels = encoded.to_numpy(dtype=np.int64)
    if schema.unlabeled_label_value is None:
        labeled_mask = np.ones(labels.shape[0], dtype=bool)
    else:
        labeled_mask = labels != schema.unlabeled_label_value

    if not labeled_mask.any():
        raise ValueError(f"{domain}: no labeled nodes are available.")
    _validate_binary(labels[labeled_mask], schema.label_column, domain)
    return labels, labeled_mask


def _load_edge_index(
    edge_path: Path,
    num_nodes: int,
    delimiter: Optional[str] = None,
    node_ids: Optional[np.ndarray] = None,
) -> Tensor:
    # Bail edge files use integer text (``0 274``), while German edge files
    # use scientific float notation (``0.0e+00 2.2e+01``) for integer node
    # indices. Parse both as floats, validate integrality, then cast.
    raw_edges = np.loadtxt(
        str(edge_path),
        dtype=np.float64,
        delimiter=delimiter,
    )
    if raw_edges.ndim == 1:
        raw_edges = raw_edges.reshape(1, -1)
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError(
            f"{edge_path} must contain exactly two integer columns per line."
        )
    if raw_edges.size == 0:
        raise ValueError(f"{edge_path} contains no edges.")
    if not np.isfinite(raw_edges).all():
        raise ValueError(f"{edge_path} contains NaN or infinity.")
    if not np.equal(raw_edges, np.floor(raw_edges)).all():
        raise ValueError(f"{edge_path} contains non-integer node endpoints.")

    edges = raw_edges.astype(np.int64)

    if node_ids is not None:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if node_ids.ndim != 1 or node_ids.shape[0] != num_nodes:
            raise ValueError(
                f"{edge_path}: node-id vector must have exactly {num_nodes} values."
            )
        if np.unique(node_ids).shape[0] != num_nodes:
            raise ValueError(
                f"{edge_path}: node IDs must be unique before edge remapping."
            )

        id_to_local = pd.Series(
            np.arange(num_nodes, dtype=np.int64),
            index=node_ids,
        )
        flat_edge_ids = pd.Series(edges.reshape(-1), copy=False)
        mapped_endpoints = flat_edge_ids.map(id_to_local)
        if mapped_endpoints.isna().any():
            missing_ids = sorted(
                {
                    int(value)
                    for value in flat_edge_ids[mapped_endpoints.isna()].unique()[:20]
                }
            )
            raise ValueError(
                f"{edge_path}: edge endpoints are absent from the domain CSV. "
                f"First missing IDs: {missing_ids}."
            )
        edges = mapped_endpoints.to_numpy(dtype=np.int64).reshape(-1, 2)

    minimum_endpoint = int(edges.min())
    maximum_endpoint = int(edges.max())
    if minimum_endpoint < 0 or maximum_endpoint >= num_nodes:
        raise ValueError(
            f"{edge_path} contains endpoint range "
            f"[{minimum_endpoint}, {maximum_endpoint}], but the feature table has "
            f"{num_nodes} rows. Edge endpoints must be local feature-row indices."
        )

    edge_index = torch.as_tensor(edges.T, dtype=torch.long).contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_remaining_self_loops(
        edge_index,
        num_nodes=num_nodes,
    )
    return edge_index.contiguous()


def _load_synthetic_domain(
    data_dir: Path,
    domain: str,
    schema: DatasetSchema,
    include_sensitive: bool,
    expected_feature_columns: Optional[Sequence[str]],
) -> DomainData:
    feature_path = data_dir / f"{domain}_feat.csv"
    label_path = data_dir / f"{domain}_label.txt"
    sensitive_path = data_dir / f"{domain}_sens.txt"
    edge_path = data_dir / f"{domain}_edges.txt"

    required_paths = [feature_path, label_path, edge_path]
    if include_sensitive:
        required_paths.append(sensitive_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"{domain}: missing required synthetic data files: {missing_paths}."
        )

    feature_frame = pd.read_csv(feature_path, header=None)
    features = feature_frame.to_numpy(dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"{domain}: synthetic feature matrix is empty or invalid.")
    if not np.isfinite(features).all():
        raise ValueError(f"{domain}: model features contain NaN or infinity.")

    available_features = [
        f"feature_{index}"
        for index in range(features.shape[1])
    ]
    if expected_feature_columns is None:
        feature_columns = available_features
    else:
        feature_columns = list(expected_feature_columns)
        if feature_columns != available_features:
            raise ValueError(
                f"{domain}: synthetic feature schema differs from the trained "
                f"model. Got {len(available_features)} ordered features, "
                f"expected {len(feature_columns)}."
            )

    label_frame = pd.read_csv(
        label_path,
        header=None,
        names=[schema.label_column],
    )
    labels, labeled_mask = _encode_label_column(
        frame=label_frame,
        schema=schema,
        domain=domain,
    )
    if labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"{domain}: feature rows ({features.shape[0]}) and labels "
            f"({labels.shape[0]}) do not match."
        )

    sensitive = None
    if include_sensitive:
        sensitive_frame = pd.read_csv(
            sensitive_path,
            header=None,
            names=[schema.sensitive_column],
        )
        sensitive = _encode_binary_column(
            frame=sensitive_frame,
            column=schema.sensitive_column,
            mapping=schema.sensitive_mapping,
            domain=domain,
        )
        if sensitive.shape[0] != features.shape[0]:
            raise ValueError(
                f"{domain}: feature rows ({features.shape[0]}) and sensitive "
                f"values ({sensitive.shape[0]}) do not match."
            )

    edge_index = _load_edge_index(
        edge_path,
        num_nodes=features.shape[0],
        delimiter=",",
    )
    return DomainData(
        domain=domain,
        features=features,
        labels=labels,
        edge_index=edge_index,
        feature_columns=feature_columns,
        schema_name=schema.name,
        id_column=schema.id_column,
        sensitive_column=schema.sensitive_column,
        label_column=schema.label_column,
        labeled_mask=labeled_mask,
        sensitive=sensitive,
    )


def load_domain(
    data_dir: Path,
    domain: str,
    include_sensitive: bool,
    expected_feature_columns: Optional[Sequence[str]] = None,
) -> DomainData:
    schema = schema_for_domain(domain)
    if schema.name == "syn":
        return _load_synthetic_domain(
            data_dir=data_dir,
            domain=domain,
            schema=schema,
            include_sensitive=include_sensitive,
            expected_feature_columns=expected_feature_columns,
        )

    csv_path = data_dir / f"{domain}.csv"
    edge_path = data_dir / f"{domain}_edges.txt"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Domain CSV not found: {csv_path}")
    if not edge_path.is_file():
        raise FileNotFoundError(f"Domain edge file not found: {edge_path}")

    frame = pd.read_csv(csv_path)
    required_columns = {
        schema.id_column,
        schema.sensitive_column,
        schema.label_column,
    }
    missing_required = sorted(required_columns.difference(frame.columns))
    if missing_required:
        raise ValueError(
            f"{domain}: missing required CSV columns {missing_required}."
        )

    excluded_feature_columns = {
        schema.id_column,
        schema.sensitive_column,
        schema.label_column,
        *schema.extra_excluded_columns,
    }
    available_features = [
        column
        for column in frame.columns
        if column not in excluded_feature_columns
    ]
    if expected_feature_columns is None:
        feature_columns = available_features
    else:
        feature_columns = list(expected_feature_columns)
        missing_features = sorted(set(feature_columns).difference(frame.columns))
        unexpected_features = sorted(
            set(available_features).difference(feature_columns)
        )
        if missing_features or unexpected_features:
            raise ValueError(
                f"{domain}: feature schema differs from the trained model. "
                f"Missing={missing_features}, unexpected={unexpected_features}."
            )

    if schema.sensitive_column in feature_columns:
        raise AssertionError(
            f"{schema.sensitive_column} must never be included in model features."
        )
    if schema.id_column in feature_columns:
        raise AssertionError(
            f"{schema.id_column} must never be included in model features."
        )
    if schema.label_column in feature_columns:
        raise AssertionError(
            f"{schema.label_column} must never be included in model features."
        )

    features = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{domain}: model features contain NaN or infinity.")

    labels, labeled_mask = _encode_label_column(
        frame=frame,
        schema=schema,
        domain=domain,
    )

    sensitive = None
    if include_sensitive:
        sensitive = _encode_binary_column(
            frame=frame,
            column=schema.sensitive_column,
            mapping=schema.sensitive_mapping,
            domain=domain,
        )

    edge_node_ids = None
    if schema.name == "pokec":
        raw_node_ids = frame[schema.id_column].to_numpy(dtype=np.float64)
        if not np.isfinite(raw_node_ids).all():
            raise ValueError(f"{domain}: user_id contains NaN or infinity.")
        if not np.equal(raw_node_ids, np.floor(raw_node_ids)).all():
            raise ValueError(f"{domain}: user_id must contain integer values.")
        edge_node_ids = raw_node_ids.astype(np.int64)

    edge_index = _load_edge_index(
        edge_path,
        num_nodes=len(frame),
        node_ids=edge_node_ids,
    )
    return DomainData(
        domain=domain,
        features=features,
        labels=labels,
        edge_index=edge_index,
        feature_columns=feature_columns,
        schema_name=schema.name,
        id_column=schema.id_column,
        sensitive_column=schema.sensitive_column,
        label_column=schema.label_column,
        labeled_mask=labeled_mask,
        sensitive=sensitive,
    )


def _checkpoint_payload(
    model: SourceOnlyGCN,
    scaler: SourceMinMaxScaler,
    source: DomainData,
    args: argparse.Namespace,
    final_train_loss: float,
) -> Dict[str, object]:
    return {
        "format_version": 1,
        "experiment": "source_only_gcn",
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
            "input_dim": int(source.features.shape[1]),
            "hidden_dim": int(args.hidden),
            "architecture": "GCNConv(input, hidden) + Linear(hidden, 1)",
            "uses_sensitive_feature": False,
        },
        "training_config": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "uses_all_available_source_labels": True,
            "source_graph_nodes": int(source.labels.shape[0]),
            "source_labeled_nodes": int(source.labeled_mask.sum()),
            "has_source_validation_split": False,
        },
        "final_train_loss": float(final_train_loss),
        "model_state_dict": model.state_dict(),
    }


def train_source(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / f"source_only_gcn_{args.source_domain}_seed{args.seed}.pt"
        )

    source_data_dir = resolve_data_dir(args.source_domain, args.data_dir)
    # Training deliberately does not materialize the sensitive attribute as a
    # vector. Its column is excluded from features and is otherwise unused.
    source = load_domain(
        data_dir=source_data_dir,
        domain=args.source_domain,
        include_sensitive=False,
    )
    scaler = SourceMinMaxScaler.fit(source.features)
    source_features = scaler.transform(source.features)

    features = torch.as_tensor(source_features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(source.labels, dtype=torch.float32, device=device)
    labeled_mask = torch.as_tensor(
        source.labeled_mask,
        dtype=torch.bool,
        device=device,
    )
    edge_index = source.edge_index.to(device)

    model = SourceOnlyGCN(
        input_dim=features.shape[1],
        hidden_dim=args.hidden,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Stage: train")
    print(f"Source domain: {source.domain}")
    print(f"Source graph nodes: {labels.numel()}")
    print(f"Source labeled nodes used for BCE: {int(labeled_mask.sum().item())}")
    print(
        "Input features after dataset-specific exclusions: "
        f"{features.shape[1]}"
    )
    print(f"Device: {device}")

    final_train_loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[labeled_mask],
            labels[labeled_mask],
        )
        loss.backward()
        optimizer.step()
        final_train_loss = float(loss.detach().cpu().item())

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d}/{args.epochs:04d} "
                f"source_bce={final_train_loss:.6f}"
            )

    checkpoint = _checkpoint_payload(
        model=model,
        scaler=scaler,
        source=source,
        args=args,
        final_train_loss=final_train_loss,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved trained source-only model: {checkpoint_path.resolve()}")


def _safe_positive_rate(
    predictions: np.ndarray,
    mask: np.ndarray,
    metric_name: str,
) -> float:
    count = int(mask.sum())
    if count == 0:
        raise ValueError(
            f"Cannot calculate {metric_name}: the required target subgroup is empty."
        )
    return float(predictions[mask].mean())


def calculate_target_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    sensitive: np.ndarray,
    sensitive_column: str,
    label_column: str,
) -> Dict[str, float]:
    predictions = (logits > 0.0).astype(np.int64)

    group_0 = sensitive == 0
    group_1 = sensitive == 1
    positive_group_0 = group_0 & (labels == 1)
    positive_group_1 = group_1 & (labels == 1)

    parity = abs(
        _safe_positive_rate(
            predictions,
            group_0,
            f"parity for {sensitive_column}=0",
        )
        - _safe_positive_rate(
            predictions,
            group_1,
            f"parity for {sensitive_column}=1",
        )
    )
    equality = abs(
        _safe_positive_rate(
            predictions,
            positive_group_0,
            f"equality for {sensitive_column}=0 and {label_column}=1",
        )
        - _safe_positive_rate(
            predictions,
            positive_group_1,
            f"equality for {sensitive_column}=1 and {label_column}=1",
        )
    )

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, logits)),
        "parity": float(parity),
        "equality": float(equality),
    }


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("experiment") != "source_only_gcn":
        raise ValueError(
            f"{checkpoint_path} is not a source_only_gcn checkpoint."
        )
    if checkpoint.get("format_version") != 1:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint.get('format_version')}."
        )
    return checkpoint


def test_target(args: argparse.Namespace) -> None:
    """Load only the trained artifact and target graph, then run inference."""

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
    # This is the only stage that materializes the sensitive attribute, and it
    # is passed only to calculate_target_metrics after model inference.
    target = load_domain(
        data_dir=target_data_dir,
        domain=args.target_domain,
        include_sensitive=True,
        expected_feature_columns=feature_columns,
    )
    if target.sensitive is None:
        raise AssertionError(
            f"Target {target.sensitive_column} values were not loaded for evaluation."
        )

    data_config = checkpoint.get("data_config")
    if data_config is not None:
        if data_config.get("schema_name") != target.schema_name:
            raise ValueError(
                "Source and target domains use incompatible dataset schemas: "
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
    target_features = scaler.transform(target.features)

    model_config = checkpoint["model_config"]
    uses_sensitive_feature = model_config.get("uses_sensitive_feature")
    # Backward compatibility for checkpoints created by the earlier Bail-only
    # implementation.
    if uses_sensitive_feature is None:
        uses_sensitive_feature = model_config.get("uses_white_feature")
    if uses_sensitive_feature is not False:
        raise ValueError(
            "Checkpoint does not certify that its sensitive attribute was excluded."
        )
    if int(model_config["input_dim"]) != target_features.shape[1]:
        raise ValueError(
            "Target feature dimension does not match the trained source model."
        )

    model = SourceOnlyGCN(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = torch.as_tensor(target_features, dtype=torch.float32, device=device)
    edge_index = target.edge_index.to(device)

    # No optimizer exists in the test process and inference_mode prevents any
    # autograd graph or parameter update.
    with torch.inference_mode():
        embedding_tensor = model.encode(features, edge_index)
        logits = model.classify(embedding_tensor).detach().cpu().numpy()
        representations = embedding_tensor.detach().cpu().numpy()

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
    visualization_method = method_name_for_seed("GCN", seed)
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
    print(f"Trained source domain: {checkpoint['source_domain']}")
    print(f"Target domain: {target.domain}")
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
            "experiment": "source_only_gcn",
            "source_domain": checkpoint["source_domain"],
            "target_domain": target.domain,
            "dataset_schema": target.schema_name,
            "label_column": target.label_column,
            "sensitive_column": target.sensitive_column,
            "checkpoint": str(args.checkpoint.resolve()),
            "target_graph_nodes": int(target.labels.shape[0]),
            "target_labeled_nodes": int(evaluation_mask.sum()),
            # Backward-compatible alias: metrics are evaluated on labeled nodes.
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


def run_source_to_target(args: argparse.Namespace) -> None:
    """Run train and test from one command while preserving process isolation.

    The wrapper itself never loads a graph. It first launches a source-only
    training child process and waits for that process to exit. Only after a
    successful checkpoint is produced does it launch a new target-only test
    process. Consequently, the test process cannot retain source tensors,
    model objects, optimizer state, or source graph data in memory.
    """

    script_path = Path(__file__).resolve()
    source_data_dir = resolve_data_dir(args.source_domain, args.source_data_dir)
    target_data_dir = resolve_data_dir(args.target_domain, args.target_data_dir)
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            PROJECT_ROOT
            / "checkpoints"
            / f"source_only_gcn_{args.source_domain}_seed{args.seed}.pt"
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
        "--hidden",
        str(args.hidden),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--log-every",
        str(args.log_every),
    ]

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

    print("Combined run: launching isolated source training process.", flush=True)
    subprocess.run(train_command, check=True)
    print(
        "Source training process exited; launching a fresh isolated target "
        "test process.",
        flush=True,
    )
    subprocess.run(test_command, check=True)


def run_multiple_seeds(args: argparse.Namespace) -> None:
    """Run isolated source-to-target experiments and aggregate their metrics."""

    seeds = list(args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicate values: {seeds}")
    if args.std_ddof < 0:
        raise ValueError("--std-ddof must be non-negative.")
    if args.std_ddof >= len(seeds):
        raise ValueError(
            f"--std-ddof must be smaller than the number of seeds "
            f"({len(seeds)}), got {args.std_ddof}."
        )

    script_path = Path(__file__).resolve()
    source_data_dir = resolve_data_dir(args.source_domain, args.source_data_dir)
    target_data_dir = resolve_data_dir(args.target_domain, args.target_data_dir)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    metric_names = ["accuracy", "roc_auc", "parity", "equality"]
    per_seed_results = []

    for run_index, seed in enumerate(seeds, start=1):
        checkpoint_path = (
            args.checkpoint_dir
            / f"source_only_gcn_{args.source_domain}_seed{seed}.pt"
        )
        result_path = (
            args.results_dir
            / (
                f"source_only_gcn_{args.source_domain}_to_"
                f"{args.target_domain}_seed{seed}.json"
            )
        )

        run_command = [
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
            "--hidden",
            str(args.hidden),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--log-every",
            str(args.log_every),
            "--visualization-embeddings-root",
            str(args.visualization_embeddings_root),
        ]

        print(
            f"\n===== Seed {seed} ({run_index}/{len(seeds)}) =====",
            flush=True,
        )
        subprocess.run(run_command, check=True)

        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        metrics = result.get("metrics", {})
        missing_metrics = [name for name in metric_names if name not in metrics]
        if missing_metrics:
            raise ValueError(
                f"{result_path} is missing target metrics: {missing_metrics}."
            )

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
                f"source_only_gcn_{args.source_domain}_to_"
                f"{args.target_domain}_seeds_{seed_label}_summary.json"
            )
        )
    else:
        summary_path = args.summary_json

    summary = {
        "experiment": "source_only_gcn_multi_seed",
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "seeds": [int(seed) for seed in seeds],
        "number_of_runs": len(seeds),
        "standard_deviation_ddof": int(args.std_ddof),
        "per_seed": per_seed_results,
        "aggregate": aggregate,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n===== Multi-seed target summary =====")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict Source-only GCN: train on one complete source graph and "
            "evaluate a frozen checkpoint on a separate target graph."
        )
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Read only the source graph, train on every source label, and save a checkpoint.",
    )
    train_parser.add_argument("--source-domain", default="bailA_2")
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Defaults to the dataset directory inferred from the domain.",
    )
    train_parser.add_argument("--checkpoint", type=Path, default=None)
    train_parser.add_argument("--hidden", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.set_defaults(handler=train_source)

    test_parser = subparsers.add_parser(
        "test",
        help="Read only a frozen checkpoint and the target graph; never update model parameters.",
    )
    test_parser.add_argument("--target-domain", default="bailA_1")
    test_parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Defaults to the dataset directory inferred from the domain.",
    )
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

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "Execute source training and frozen target testing from one command "
            "using two isolated child processes."
        ),
    )
    run_parser.add_argument("--source-domain", default="bailA_2")
    run_parser.add_argument("--target-domain", default="bailA_1")
    run_parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=None,
        help="Defaults to the dataset directory inferred from --source-domain.",
    )
    run_parser.add_argument(
        "--target-data-dir",
        type=Path,
        default=None,
        help="Defaults to the dataset directory inferred from --target-domain.",
    )
    run_parser.add_argument("--checkpoint", type=Path, default=None)
    run_parser.add_argument("--output-json", type=Path, default=None)
    run_parser.add_argument(
        "--visualization-embeddings-root",
        type=Path,
        default=DEFAULT_VISUALIZATION_EMBEDDINGS_ROOT,
    )
    run_parser.add_argument("--hidden", type=int, default=16)
    run_parser.add_argument("--epochs", type=int, default=1000)
    run_parser.add_argument("--lr", type=float, default=1e-3)
    run_parser.add_argument("--weight-decay", type=float, default=1e-5)
    run_parser.add_argument("--seed", type=int, default=1)
    run_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run_parser.add_argument("--log-every", type=int, default=100)
    run_parser.set_defaults(handler=run_source_to_target)

    multi_seed_parser = subparsers.add_parser(
        "run-seeds",
        help=(
            "Run seeds 1--5 by default, keeping every train/test pair process-"
            "isolated, then aggregate target metrics."
        ),
    )
    multi_seed_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
    )
    multi_seed_parser.add_argument("--source-domain", default="bailA_2")
    multi_seed_parser.add_argument("--target-domain", default="bailA_1")
    multi_seed_parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=None,
        help="Defaults to the dataset directory inferred from --source-domain.",
    )
    multi_seed_parser.add_argument(
        "--target-data-dir",
        type=Path,
        default=None,
        help="Defaults to the dataset directory inferred from --target-domain.",
    )
    multi_seed_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    multi_seed_parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    multi_seed_parser.add_argument("--summary-json", type=Path, default=None)
    multi_seed_parser.add_argument(
        "--visualization-embeddings-root",
        type=Path,
        default=DEFAULT_VISUALIZATION_EMBEDDINGS_ROOT,
    )
    multi_seed_parser.add_argument(
        "--std-ddof",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Standard-deviation convention: 0 for population std (NumPy "
            "default), 1 for sample std."
        ),
    )
    multi_seed_parser.add_argument("--hidden", type=int, default=16)
    multi_seed_parser.add_argument("--epochs", type=int, default=1000)
    multi_seed_parser.add_argument("--lr", type=float, default=1e-3)
    multi_seed_parser.add_argument("--weight-decay", type=float, default=1e-5)
    multi_seed_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    multi_seed_parser.add_argument("--log-every", type=int, default=100)
    multi_seed_parser.set_defaults(handler=run_multiple_seeds)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
