"""Source-only FairSIN transfer for BailA, GermanA, Pokec, and Synthetic splits.

The source graph is used for every trainable decision: feature preprocessing,
train/validation splitting, heterogeneous-neighbor regression, and adversarial
training. Every run always completes the configured number of epochs and uses
its final model state. Only after all source runs finish and source tensors are
released is the target graph loaded for frozen, inference-only evaluation.

This is the full model-centric FairSIN variant used by this repository:

* an MLP estimates a node's heterogeneous-neighbor feature (F3);
* F3 is added before GNN message passing;
* a sensitive-attribute discriminator is trained adversarially; and
* a task classifier predicts the dataset-specific downstream label.
"""

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_scipy_sparse_matrix

from dataset import sparse_mx_to_torch_sparse_tensor, sys_normalized_adjacency
from model import (
    GCN_encoder_scatter,
    GCN_encoder_spmm,
    GIN_encoder,
    MLP_classifier,
    MLP_discriminator,
    SAGE_encoder,
)
from utils import seed_everything


@dataclass(frozen=True)
class DatasetProfile:
    """Dataset-specific schema and preprocessing used by the original code."""

    family: str
    domain_prefix: str
    data_dir: str
    default_source: str
    default_target: str
    label_column: str
    sensitive_column: str
    id_column: str
    drop_columns: Tuple[str, ...]
    label_mapping: Dict[object, int]
    sensitive_mapping: Dict[object, int]
    normalization: str
    label_mode: str
    domain_subdirectory: bool
    edge_index_mode: str
    feature_alignment: str
    storage_format: str = "table"
    edge_delimiter: Optional[str] = None


DATASET_PROFILES = {
    "bailA": DatasetProfile(
        family="bailA",
        domain_prefix="bailA_",
        data_dir="dataset/bailA",
        default_source="bailA_2",
        default_target="bailA_1",
        label_column="RECID",
        sensitive_column="WHITE",
        id_column="user_id",
        drop_columns=(),
        label_mapping={0: 0, 1: 1},
        sensitive_mapping={0: 0, 1: 1},
        normalization="source-minmax",
        label_mode="mapping",
        domain_subdirectory=False,
        edge_index_mode="row",
        feature_alignment="exact",
    ),
    "germanA": DatasetProfile(
        family="germanA",
        domain_prefix="germanA_",
        data_dir="dataset/germanA",
        default_source="germanA_2",
        default_target="germanA_1",
        label_column="GoodCustomer",
        sensitive_column="Gender",
        id_column="user_id",
        # These are the same two columns removed by load_german/load_germanA
        # in dataset.py. PurposeOfLoan is categorical; OtherLoansAtStore is
        # intentionally excluded to preserve the original FairSIN protocol.
        drop_columns=("PurposeOfLoan", "OtherLoansAtStore"),
        label_mapping={-1: 0, 1: 1},
        sensitive_mapping={"Male": 0, "Female": 1},
        # The original repository explicitly skips feature_norm for German.
        normalization="none",
        label_mode="mapping",
        domain_subdirectory=False,
        edge_index_mode="row",
        feature_alignment="exact",
    ),
    "pokec": DatasetProfile(
        family="pokec",
        domain_prefix="pokec_",
        data_dir="dataset",
        default_source="pokec_z",
        default_target="pokec_n",
        label_column="I_am_working_in_field",
        sensitive_column="region",
        id_column="user_id",
        drop_columns=(),
        label_mapping={},
        sensitive_mapping={0: 0, 1: 1},
        normalization="source-minmax",
        label_mode="positive-if-greater-zero",
        domain_subdirectory=True,
        edge_index_mode="user_id",
        feature_alignment="intersection",
    ),
    "syn": DatasetProfile(
        family="syn",
        domain_prefix="syn-",
        data_dir="dataset/syn",
        default_source="syn-2",
        default_target="syn-1",
        # Synthetic labels and sensitive attributes live in separate files;
        # these names define their model-facing schema.
        label_column="label",
        sensitive_column="sensitive",
        id_column="",
        drop_columns=(),
        label_mapping={0: 0, 1: 1},
        sensitive_mapping={0: 0, 1: 1},
        normalization="source-minmax",
        label_mode="mapping",
        domain_subdirectory=False,
        edge_index_mode="row",
        feature_alignment="exact",
        storage_format="synthetic",
        edge_delimiter=",",
    ),
}


@dataclass
class GraphDomain:
    """One graph domain with a node space independent of every other domain."""

    name: str
    x: torch.Tensor
    y: torch.Tensor
    label_mask: torch.Tensor
    sens: torch.Tensor
    edge_index: torch.Tensor
    adj_norm_sp: torch.Tensor
    adj: sp.csr_matrix
    feature_names: Tuple[str, ...]

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def to(self, device: torch.device) -> "GraphDomain":
        return GraphDomain(
            name=self.name,
            x=self.x.to(device),
            y=self.y.to(device),
            label_mask=self.label_mask.to(device),
            sens=self.sens.to(device),
            edge_index=self.edge_index.to(device),
            adj_norm_sp=self.adj_norm_sp.coalesce().to(device),
            adj=self.adj,
            feature_names=self.feature_names,
        )


@dataclass(frozen=True)
class SourcePreprocessor:
    """Source-fitted preprocessing state that contains no source node rows."""

    feature_names: Tuple[str, ...]
    sensitive_column: str
    sensitive_index: int
    normalization: str
    source_min: torch.Tensor
    source_max: torch.Tensor
    constant_columns: torch.Tensor


@dataclass
class TrainedRunArtifact:
    """Frozen final-epoch weights passed across the train/test boundary."""

    run_index: int
    neutralizer_state: Dict[str, torch.Tensor]
    encoder_state: Dict[str, torch.Tensor]
    classifier_state: Dict[str, torch.Tensor]


@dataclass
class TransferMetrics:
    accuracy: float
    auc_roc: float
    f1: float
    demographic_parity: float
    equal_opportunity: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "f1": self.f1,
            "demographic_parity": self.demographic_parity,
            "equal_opportunity": self.equal_opportunity,
        }


class FeatureNeutralizer(nn.Module):
    """Three-layer F3 estimator used in the original FairSIN implementation."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def domain_file_paths(
    data_dir: Path,
    domain_name: str,
    domain_subdirectory: bool,
) -> Tuple[Path, Path]:
    domain_dir = data_dir / domain_name if domain_subdirectory else data_dir
    return (
        domain_dir / "{}.csv".format(domain_name),
        domain_dir / "{}_edges.txt".format(domain_name),
    )


def synthetic_file_paths(
    data_dir: Path,
    domain_name: str,
) -> Tuple[Path, Path, Path, Path]:
    """Return feature, label, sensitive, and edge paths for Synthetic data."""

    return (
        data_dir / "{}_feat.csv".format(domain_name),
        data_dir / "{}_label.txt".format(domain_name),
        data_dir / "{}_sens.txt".format(domain_name),
        data_dir / "{}_edges.txt".format(domain_name),
    )


def resolve_aligned_feature_names(
    data_dir: Path,
    source_name: str,
    target_name: str,
    profile: DatasetProfile,
    label_column: str,
    sensitive_column: str,
    id_column: str,
    include_id_feature: bool,
    drop_columns: Tuple[str, ...],
) -> Tuple[Tuple[str, ...], Dict[str, object]]:
    """Resolve a shared model schema without loading either complete domain."""

    if profile.storage_format == "synthetic":
        if include_id_feature:
            raise ValueError(
                "Synthetic domains have no identifier column; "
                "--include-id-feature is not supported"
            )
        source_feature_path, _, _, _ = synthetic_file_paths(data_dir, source_name)
        target_feature_path, _, _, _ = synthetic_file_paths(data_dir, target_name)
        for feature_path in (source_feature_path, target_feature_path):
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)

        # Synthetic feature files have no header. Inspect one feature row from
        # each domain only to fix and validate the shared dimensional schema.
        source_width = pd.read_csv(
            source_feature_path, header=None, nrows=1
        ).shape[1]
        target_width = pd.read_csv(
            target_feature_path, header=None, nrows=1
        ).shape[1]
        if source_width <= 0 or target_width <= 0:
            raise ValueError("Synthetic feature files must contain at least one column")
        if source_width != target_width:
            raise ValueError(
                "Source and target Synthetic feature widths differ: "
                "source={} target={}".format(source_width, target_width)
            )

        full_features = tuple(
            "feature_{:03d}".format(index) for index in range(source_width)
        ) + (sensitive_column,)
        unknown_drops = set(drop_columns).difference(full_features)
        if unknown_drops:
            raise ValueError(
                "Unknown Synthetic feature columns requested for removal: {}".format(
                    sorted(unknown_drops)
                )
            )
        aligned = tuple(
            feature for feature in full_features if feature not in set(drop_columns)
        )
        if sensitive_column not in aligned:
            raise ValueError(
                "The Synthetic sensitive feature must remain in the model input"
            )
        return aligned, {
            "mode": profile.feature_alignment,
            "source_feature_count": len(full_features),
            "target_feature_count": len(full_features),
            "aligned_feature_count": len(aligned),
            "source_only_features": (),
            "target_only_features": (),
            "target_header_only_inspected_before_training": False,
            "target_schema_row_only_inspected_before_training": True,
            "synthetic_raw_feature_count": source_width,
            "sensitive_feature_appended": True,
        }

    if profile.storage_format != "table":
        raise ValueError(
            "Unsupported dataset storage format: {}".format(profile.storage_format)
        )

    source_csv, _ = domain_file_paths(
        data_dir, source_name, profile.domain_subdirectory
    )
    target_csv, _ = domain_file_paths(
        data_dir, target_name, profile.domain_subdirectory
    )
    source_columns = tuple(pd.read_csv(source_csv, nrows=0).columns)
    target_columns = tuple(pd.read_csv(target_csv, nrows=0).columns)
    excluded = {label_column}.union(drop_columns)
    if not include_id_feature:
        excluded.add(id_column)
    source_features = tuple(
        column for column in source_columns if column not in excluded
    )
    target_features = tuple(
        column for column in target_columns if column not in excluded
    )

    if profile.feature_alignment == "exact":
        if source_features != target_features:
            raise ValueError(
                "Source and target feature columns differ for exact alignment:\n"
                "source={}\ntarget={}".format(source_features, target_features)
            )
        aligned = source_features
    elif profile.feature_alignment == "intersection":
        target_set = set(target_features)
        aligned = tuple(column for column in source_features if column in target_set)
        if not aligned:
            raise ValueError("Source and target have no common model features")
    else:
        raise ValueError(
            "Unsupported feature alignment mode: {}".format(
                profile.feature_alignment
            )
        )

    alignment_info = {
        "mode": profile.feature_alignment,
        "source_feature_count": len(source_features),
        "target_feature_count": len(target_features),
        "aligned_feature_count": len(aligned),
        "source_only_features": tuple(
            column for column in source_features if column not in set(target_features)
        ),
        "target_only_features": tuple(
            column for column in target_features if column not in set(source_features)
        ),
        "target_header_only_inspected_before_training": True,
    }
    return aligned, alignment_info


def _load_edges(
    edge_path: Path,
    num_nodes: int,
    node_ids: np.ndarray,
    edge_index_mode: str,
    edge_delimiter: Optional[str] = None,
) -> sp.csr_matrix:
    """Load edges, map endpoints to rows, symmetrize, and remove self-loops."""

    raw_edges = np.genfromtxt(
        str(edge_path), dtype=np.float64, delimiter=edge_delimiter
    )
    if raw_edges.size == 0:
        raise ValueError("Edge file is empty: {}".format(edge_path))
    if raw_edges.ndim == 1:
        raw_edges = raw_edges.reshape(1, -1)
    if raw_edges.shape[1] != 2:
        raise ValueError(
            "Expected two columns in {}, got {}".format(edge_path, raw_edges.shape)
        )
    if not np.isfinite(raw_edges).all():
        raise ValueError("{} contains a non-finite edge endpoint".format(edge_path))
    rounded_edges = np.rint(raw_edges)
    if not np.allclose(raw_edges, rounded_edges):
        raise ValueError("{} contains a non-integer edge endpoint".format(edge_path))
    edges = rounded_edges.astype(np.int64)
    if edge_index_mode == "row":
        if edges.min() < 0 or edges.max() >= num_nodes:
            raise ValueError(
                "{} contains an endpoint outside [0, {}]. This profile expects "
                "CSV row positions.".format(edge_path, num_nodes - 1)
            )
    elif edge_index_mode == "user_id":
        node_index = pd.Index(node_ids.astype(np.int64))
        if not node_index.is_unique:
            raise ValueError("Node identifiers are not unique for {}".format(edge_path))
        mapped = node_index.get_indexer(edges.reshape(-1))
        if (mapped < 0).any():
            unknown = np.unique(edges.reshape(-1)[mapped < 0])[:10].tolist()
            raise ValueError(
                "{} contains endpoints missing from the CSV user_id column; "
                "examples={}".format(edge_path, unknown)
            )
        edges = mapped.reshape(edges.shape).astype(np.int64)
    else:
        raise ValueError("Unsupported edge index mode: {}".format(edge_index_mode))

    values = np.ones(edges.shape[0], dtype=np.float32)
    adj = sp.coo_matrix(
        (values, (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    ).tocsr()
    adj.data[:] = 1.0
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj


def _encode_binary_column(
    values: pd.Series,
    mapping: Dict[object, int],
    column_name: str,
    csv_path: Path,
) -> pd.Series:
    encoded = values.map(mapping)
    if encoded.isna().any():
        unknown_values = sorted(
            values.loc[encoded.isna()].astype(str).unique().tolist()
        )
        raise ValueError(
            "{} contains unsupported {} values: {}".format(
                csv_path, column_name, unknown_values
            )
        )
    return encoded.astype(np.int64)


def _encode_label_column(
    values: pd.Series,
    profile: DatasetProfile,
    column_name: str,
    csv_path: Path,
) -> Tuple[pd.Series, pd.Series]:
    if profile.label_mode == "mapping":
        encoded = _encode_binary_column(
            values, profile.label_mapping, column_name, csv_path
        )
        label_mask = pd.Series(True, index=values.index)
    elif profile.label_mode == "positive-if-greater-zero":
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any():
            unknown = values.loc[numeric.isna()].astype(str).unique()[:10].tolist()
            raise ValueError(
                "{} contains non-numeric {} values: {}".format(
                    csv_path, column_name, unknown
                )
            )
        label_mask = numeric >= 0
        encoded = (numeric > 0).astype(np.int64)
    else:
        raise ValueError("Unsupported label mode: {}".format(profile.label_mode))
    return encoded, label_mask.astype(bool)


def load_synthetic_domain(
    data_dir: Path,
    domain_name: str,
    profile: DatasetProfile,
    label_column: str,
    sensitive_column: str,
    feature_names: Tuple[str, ...],
) -> GraphDomain:
    """Load a Synthetic domain whose node fields are stored separately."""

    feature_path, label_path, sensitive_path, edge_path = synthetic_file_paths(
        data_dir, domain_name
    )
    for required_path in (feature_path, label_path, sensitive_path, edge_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    feature_frame = pd.read_csv(feature_path, header=None)
    label_frame = pd.read_csv(label_path, header=None)
    sensitive_frame = pd.read_csv(sensitive_path, header=None)
    if feature_frame.empty:
        raise ValueError("Synthetic feature file is empty: {}".format(feature_path))
    if label_frame.shape[1] != 1:
        raise ValueError(
            "Expected one label column in {}, got {}".format(
                label_path, label_frame.shape
            )
        )
    if sensitive_frame.shape[1] != 1:
        raise ValueError(
            "Expected one sensitive column in {}, got {}".format(
                sensitive_path, sensitive_frame.shape
            )
        )

    num_nodes = len(feature_frame)
    if len(label_frame) != num_nodes or len(sensitive_frame) != num_nodes:
        raise ValueError(
            "Synthetic row counts differ for {}: features={} labels={} "
            "sensitive={}".format(
                domain_name, num_nodes, len(label_frame), len(sensitive_frame)
            )
        )
    try:
        raw_features = feature_frame.to_numpy(dtype=np.float32)
    except ValueError as error:
        raise ValueError(
            "{} contains a non-numeric Synthetic feature".format(feature_path)
        ) from error
    if not np.isfinite(raw_features).all():
        raise ValueError("{} contains non-finite model features".format(feature_path))

    encoded_labels, label_mask = _encode_label_column(
        label_frame.iloc[:, 0], profile, label_column, label_path
    )
    encoded_sensitive = _encode_binary_column(
        sensitive_frame.iloc[:, 0],
        profile.sensitive_mapping,
        sensitive_column,
        sensitive_path,
    )
    full_feature_names = tuple(
        "feature_{:03d}".format(index) for index in range(raw_features.shape[1])
    ) + (sensitive_column,)
    missing_features = set(feature_names).difference(full_feature_names)
    if missing_features:
        raise ValueError(
            "{} cannot provide requested Synthetic features: {}".format(
                feature_path, sorted(missing_features)
            )
        )
    full_feature_values = np.concatenate(
        [
            raw_features,
            encoded_sensitive.to_numpy(dtype=np.float32)[:, None],
        ],
        axis=1,
    )
    selected_indices = [full_feature_names.index(name) for name in feature_names]
    feature_values = full_feature_values[:, selected_indices]

    x = torch.tensor(feature_values, dtype=torch.float32)
    y = torch.tensor(encoded_labels.to_numpy(dtype=np.float32))
    label_mask_tensor = torch.tensor(label_mask.to_numpy(dtype=bool))
    sens = torch.tensor(encoded_sensitive.to_numpy(dtype=np.int64))
    node_ids = np.arange(num_nodes, dtype=np.int64)

    label_values = set(np.unique(y.numpy()).tolist())
    sensitive_values = set(np.unique(sens.numpy()).tolist())
    if not label_values.issubset({0.0, 1.0}):
        raise ValueError("{} must be binary; found {}".format(label_column, label_values))
    if not sensitive_values.issubset({0, 1}):
        raise ValueError(
            "{} must be binary for this FairSIN implementation; found {}".format(
                sensitive_column, sensitive_values
            )
        )

    adj = _load_edges(
        edge_path=edge_path,
        num_nodes=num_nodes,
        node_ids=node_ids,
        edge_index_mode=profile.edge_index_mode,
        edge_delimiter=profile.edge_delimiter,
    )
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    return GraphDomain(
        name=domain_name,
        x=x,
        y=y,
        label_mask=label_mask_tensor,
        sens=sens,
        edge_index=edge_index.long(),
        adj_norm_sp=adj_norm_sp,
        adj=adj,
        feature_names=feature_names,
    )


def load_graph_domain(
    data_dir: Path,
    domain_name: str,
    profile: DatasetProfile,
    label_column: str,
    sensitive_column: str,
    id_column: str,
    feature_names: Tuple[str, ...],
) -> GraphDomain:
    """Load one independent graph domain using a resolved dataset profile."""

    if profile.storage_format == "synthetic":
        return load_synthetic_domain(
            data_dir=data_dir,
            domain_name=domain_name,
            profile=profile,
            label_column=label_column,
            sensitive_column=sensitive_column,
            feature_names=feature_names,
        )
    if profile.storage_format != "table":
        raise ValueError(
            "Unsupported dataset storage format: {}".format(profile.storage_format)
        )

    csv_path, edge_path = domain_file_paths(
        data_dir, domain_name, profile.domain_subdirectory
    )
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not edge_path.is_file():
        raise FileNotFoundError(edge_path)

    frame = pd.read_csv(csv_path)
    required = {label_column, sensitive_column, id_column}.union(feature_names)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("{} is missing columns: {}".format(csv_path, sorted(missing)))

    frame[label_column], label_mask = _encode_label_column(
        frame[label_column], profile, label_column, csv_path
    )
    frame[sensitive_column] = _encode_binary_column(
        frame[sensitive_column],
        profile.sensitive_mapping,
        sensitive_column,
        csv_path,
    )

    try:
        feature_values = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
    except ValueError as error:
        raise ValueError(
            "Non-numeric model feature remains in {} after preprocessing. "
            "Feature columns: {}".format(csv_path, feature_names)
        ) from error
    if not np.isfinite(feature_values).all():
        raise ValueError("{} contains non-finite model features".format(csv_path))

    x = torch.tensor(feature_values)
    y = torch.tensor(frame[label_column].to_numpy(dtype=np.float32))
    label_mask_tensor = torch.tensor(label_mask.to_numpy(dtype=bool))
    sens = torch.tensor(frame[sensitive_column].to_numpy(dtype=np.int64))
    node_ids = frame[id_column].to_numpy(dtype=np.int64)

    label_values = set(np.unique(y.numpy()).tolist())
    sensitive_values = set(np.unique(sens.numpy()).tolist())
    if not label_values.issubset({0.0, 1.0}):
        raise ValueError("{} must be binary; found {}".format(label_column, label_values))
    if not sensitive_values.issubset({0, 1}):
        raise ValueError(
            "{} must be binary for this FairSIN implementation; found {}".format(
                sensitive_column, sensitive_values
            )
        )

    adj = _load_edges(
        edge_path=edge_path,
        num_nodes=len(frame),
        node_ids=node_ids,
        edge_index_mode=profile.edge_index_mode,
        edge_delimiter=profile.edge_delimiter,
    )
    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    return GraphDomain(
        name=domain_name,
        x=x,
        y=y,
        label_mask=label_mask_tensor,
        sens=sens,
        edge_index=edge_index.long(),
        adj_norm_sp=adj_norm_sp,
        adj=adj,
        feature_names=feature_names,
    )


def fit_source_preprocessor(
    source: GraphDomain,
    sensitive_column: str,
    normalization: str,
) -> Tuple[GraphDomain, SourcePreprocessor]:
    """Fit preprocessing on source only and immediately transform the source."""

    if sensitive_column not in source.feature_names:
        raise ValueError("Sensitive column is not present in the model input")
    if normalization not in {"source-minmax", "none"}:
        raise ValueError("Unsupported normalization mode: {}".format(normalization))

    sensitive_index = source.feature_names.index(sensitive_column)
    source_min = source.x.min(dim=0).values
    source_max = source.x.max(dim=0).values
    preprocessor = SourcePreprocessor(
        feature_names=source.feature_names,
        sensitive_column=sensitive_column,
        sensitive_index=sensitive_index,
        normalization=normalization,
        source_min=source_min,
        source_max=source_max,
        constant_columns=(source_max - source_min) == 0,
    )
    source = apply_source_preprocessor(source, preprocessor)
    return source, preprocessor


def apply_source_preprocessor(
    domain: GraphDomain,
    preprocessor: SourcePreprocessor,
) -> GraphDomain:
    """Apply frozen source preprocessing parameters to one graph domain."""

    if domain.feature_names != preprocessor.feature_names:
        raise ValueError(
            "Feature columns differ from the source training schema:\n"
            "source={}\ncurrent={}".format(
                preprocessor.feature_names, domain.feature_names
            )
        )

    if preprocessor.normalization == "source-minmax":
        source_range = preprocessor.source_max - preprocessor.source_min
        safe_range = source_range.clone()
        safe_range[preprocessor.constant_columns] = 1.0
        original_sensitive = domain.x[:, preprocessor.sensitive_index].clone()
        domain.x = 2.0 * (domain.x - preprocessor.source_min) / safe_range - 1.0
        domain.x[:, preprocessor.constant_columns] = 0.0
        # Match the original repository preprocessing: keep the sensitive
        # feature binary instead of mapping it into the normalized range.
        domain.x[:, preprocessor.sensitive_index] = original_sensitive
    return domain


def count_values_outside_source_range(
    domain: GraphDomain,
    preprocessor: SourcePreprocessor,
) -> int:
    """Count target covariates outside source-observed ranges before transform."""

    outside = (
        (domain.x < preprocessor.source_min)
        | (domain.x > preprocessor.source_max)
    )
    outside[:, preprocessor.sensitive_index] = False
    return int(outside.sum().item())


def heterogeneous_neighbor_targets(
    source: GraphDomain,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Compute F3 regression targets from source-domain neighbors only."""

    coo = source.adj.tocoo()
    sens = source.sens.cpu().numpy()
    is_heterogeneous = sens[coo.row] != sens[coo.col]
    row = coo.row[is_heterogeneous]
    col = coo.col[is_heterogeneous]
    values = np.ones(len(row), dtype=np.float32)
    hetero_adj = sp.csr_matrix(
        (values, (row, col)),
        shape=source.adj.shape,
        dtype=np.float32,
    )

    degree = np.asarray(hetero_adj.sum(axis=1)).reshape(-1)
    has_heterogeneous_neighbor = degree > 0
    feature_sum = hetero_adj.dot(source.x.cpu().numpy())
    feature_mean = np.zeros_like(feature_sum, dtype=np.float32)
    feature_mean[has_heterogeneous_neighbor] = (
        feature_sum[has_heterogeneous_neighbor]
        / degree[has_heterogeneous_neighbor, None]
    )
    return (
        torch.from_numpy(feature_mean),
        torch.from_numpy(has_heterogeneous_neighbor),
        degree,
    )


def source_train_validation_masks(
    source: GraphDomain,
    validation_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hold out a source-only validation set, stratified by label and group."""

    if validation_ratio <= 0.0 or validation_ratio >= 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    eligible = source.label_mask.cpu().numpy().astype(bool)
    indices = np.where(eligible)[0]
    if indices.size == 0:
        raise ValueError("Source domain has no labeled nodes")
    y = source.y.cpu().numpy().astype(np.int64)[indices]
    sens = source.sens.cpu().numpy().astype(np.int64)[indices]
    joint_stratum = 2 * y + sens
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
        stratify=joint_stratum,
    )

    train_mask = torch.zeros(source.num_nodes, dtype=torch.bool)
    validation_mask = torch.zeros(source.num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(train_indices)] = True
    validation_mask[torch.from_numpy(validation_indices)] = True
    return train_mask, validation_mask


def build_encoder(args: argparse.Namespace) -> nn.Module:
    if args.encoder == "GCN":
        if args.prop == "scatter":
            return GCN_encoder_scatter(args)
        return GCN_encoder_spmm(args)
    if args.encoder == "GIN":
        return GIN_encoder(args)
    if args.encoder == "SAGE":
        return SAGE_encoder(args)
    raise ValueError("Unsupported encoder: {}".format(args.encoder))


def forward_model(
    domain: GraphDomain,
    neutralizer: FeatureNeutralizer,
    encoder: nn.Module,
    classifier: MLP_classifier,
    delta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_heterogeneous_feature = neutralizer(domain.x)
    neutralized_feature = domain.x + delta * predicted_heterogeneous_feature
    embedding = encoder(
        neutralized_feature,
        domain.edge_index,
        domain.adj_norm_sp,
    )
    logits = classifier(embedding).view(-1)
    return logits, embedding, predicted_heterogeneous_feature


def _safe_group_fairness(
    prediction: np.ndarray,
    labels: np.ndarray,
    sensitive: np.ndarray,
) -> Tuple[float, float]:
    group_zero = sensitive == 0
    group_one = sensitive == 1
    if not group_zero.any() or not group_one.any():
        demographic_parity = float("nan")
    else:
        demographic_parity = float(
            abs(prediction[group_zero].mean() - prediction[group_one].mean())
        )

    group_zero_positive = group_zero & (labels == 1)
    group_one_positive = group_one & (labels == 1)
    if not group_zero_positive.any() or not group_one_positive.any():
        equal_opportunity = float("nan")
    else:
        equal_opportunity = float(
            abs(
                prediction[group_zero_positive].mean()
                - prediction[group_one_positive].mean()
            )
        )
    return demographic_parity, equal_opportunity


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    mask: torch.Tensor,
) -> TransferMetrics:
    selected_logits = logits[mask].detach().cpu().numpy()
    selected_labels = labels[mask].detach().cpu().numpy().astype(np.int64)
    selected_sensitive = sensitive[mask].detach().cpu().numpy().astype(np.int64)
    prediction = (selected_logits > 0.0).astype(np.int64)

    accuracy = float((prediction == selected_labels).mean())
    f1 = float(f1_score(selected_labels, prediction, zero_division=0))
    if np.unique(selected_labels).size < 2:
        auc_roc = float("nan")
    else:
        auc_roc = float(roc_auc_score(selected_labels, selected_logits))
    demographic_parity, equal_opportunity = _safe_group_fairness(
        prediction, selected_labels, selected_sensitive
    )
    return TransferMetrics(
        accuracy=accuracy,
        auc_roc=auc_roc,
        f1=f1,
        demographic_parity=demographic_parity,
        equal_opportunity=equal_opportunity,
    )


def validation_tradeoff(metrics: TransferMetrics, alpha: float) -> float:
    auc = metrics.auc_roc if math.isfinite(metrics.auc_roc) else 0.5
    dp = metrics.demographic_parity if math.isfinite(metrics.demographic_parity) else 1.0
    eo = metrics.equal_opportunity if math.isfinite(metrics.equal_opportunity) else 1.0
    return metrics.accuracy + metrics.f1 + auc - alpha * (dp + eo)


def state_dict_on_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def format_metrics(metrics: TransferMetrics) -> str:
    return (
        "ACC={:.4f} AUC={:.4f} F1={:.4f} DP={:.4f} EO={:.4f}".format(
            metrics.accuracy,
            metrics.auc_roc,
            metrics.f1,
            metrics.demographic_parity,
            metrics.equal_opportunity,
        )
    )


def encode_visualization_groups(
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Encode target ``(Y, S)`` pairs using the visualization convention.

    The zyt visualization reader uses the fixed mapping:
    ``0 -> Y=1,S=0``, ``1 -> Y=1,S=1``, ``2 -> Y=0,S=0``,
    and ``3 -> Y=0,S=1``.  Only valid/labeled target rows selected by ``mask``
    are exported.
    """

    selected_y = labels[mask].detach().to(dtype=torch.long)
    selected_sensitive = sensitive[mask].detach().to(dtype=torch.long)
    if not torch.all((selected_y == 0) | (selected_y == 1)):
        raise ValueError("Visualization target labels must be binary 0/1")
    if not torch.all((selected_sensitive == 0) | (selected_sensitive == 1)):
        raise ValueError("Visualization sensitive attributes must be binary 0/1")
    # Y=1 groups come first (0/1), followed by Y=0 groups (2/3).
    groups = (1 - selected_y) * 2 + selected_sensitive
    return groups.cpu().numpy().astype(np.int64, copy=False)


def save_visualization_embeddings_current_folder(
    representations: np.ndarray,
    labels: np.ndarray,
    run_index: int,
) -> Tuple[Path, Path]:
    """Save the standard visualization NPZ pair directly in the CWD.

    The output location is deliberately ``Path.cwd()`` rather than a path
    derived from ``__file__`` or its parent.  With ``runs > 1`` the canonical
    pair is refreshed after every seed, so the files contain the most recently
    completed run; use separate ``--runs 1 --seed ...`` invocations when each
    seed must be archived under the exact canonical filenames.
    """

    embeddings = np.asarray(representations, dtype=np.float32)
    group_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if embeddings.ndim != 2:
        raise ValueError(
            "Visualization representations must be 2D, got {}".format(
                embeddings.shape
            )
        )
    if embeddings.shape[0] != group_labels.shape[0]:
        raise ValueError(
            "Visualization representations/labels length mismatch: {} vs {}".format(
                embeddings.shape[0], group_labels.shape[0]
            )
        )
    if not np.isfinite(embeddings).all():
        raise FloatingPointError(
            "Visualization representations contain NaN or Inf; refusing to save"
        )
    if not np.isin(group_labels, np.asarray([0, 1, 2, 3], dtype=np.int64)).all():
        raise ValueError("Visualization labels must contain only 0, 1, 2, or 3")

    output_dir = Path.cwd()
    feature_path = output_dir / "feat.npz"
    labels_path = output_dir / "labels.npz"
    np.savez_compressed(feature_path, representations=embeddings)
    np.savez_compressed(labels_path, labels=group_labels)
    print(
        "run={} VISUALIZATION EXPORT feat={} labels={} shape={} labels_shape={}".format(
            run_index + 1,
            feature_path,
            labels_path,
            tuple(embeddings.shape),
            tuple(group_labels.shape),
        )
    )
    return feature_path, labels_path


def save_checkpoint(
    checkpoint_path: Path,
    neutralizer: FeatureNeutralizer,
    encoder: nn.Module,
    classifier: MLP_classifier,
    discriminator: MLP_discriminator,
    args: argparse.Namespace,
    source_validation: TransferMetrics,
    preprocessor: SourcePreprocessor,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_args = vars(args).copy()
    checkpoint_args["device"] = str(checkpoint_args["device"])
    torch.save(
        {
            "neutralizer": state_dict_on_cpu(neutralizer),
            "encoder": state_dict_on_cpu(encoder),
            "classifier": state_dict_on_cpu(classifier),
            "discriminator": state_dict_on_cpu(discriminator),
            "final_source_validation": source_validation.as_dict(),
            "source_min": preprocessor.source_min.cpu(),
            "source_max": preprocessor.source_max.cpu(),
            "constant_columns": preprocessor.constant_columns.cpu(),
            "feature_names": preprocessor.feature_names,
            "sensitive_column": preprocessor.sensitive_column,
            "sensitive_index": preprocessor.sensitive_index,
            "normalization": preprocessor.normalization,
            "args": checkpoint_args,
            "selection_policy": "fixed_epochs_final_state",
            "training_epochs_completed": args.epochs,
        },
        checkpoint_path,
    )


def train_source_once(
    source_cpu: GraphDomain,
    heterogeneous_mean_cpu: torch.Tensor,
    has_heterogeneous_neighbor_cpu: torch.Tensor,
    preprocessor: SourcePreprocessor,
    args: argparse.Namespace,
    run_index: int,
) -> TrainedRunArtifact:
    """Train exactly args.epochs on source and return frozen final weights."""

    run_seed = args.seed + run_index
    seed_everything(run_seed)
    source_train_mask, source_validation_mask = source_train_validation_masks(
        source_cpu, args.validation_ratio, run_seed
    )
    predictor_train_mask = (
        source_train_mask | ~source_cpu.label_mask
    ) & has_heterogeneous_neighbor_cpu
    predictor_validation_mask = source_validation_mask & has_heterogeneous_neighbor_cpu
    fairness_train_mask = source_train_mask | ~source_cpu.label_mask
    if not predictor_train_mask.any():
        raise RuntimeError("No source training node has a heterogeneous neighbor")

    device = args.device
    source = source_cpu.to(device)
    heterogeneous_mean = heterogeneous_mean_cpu.to(device)
    source_train_mask = source_train_mask.to(device)
    source_validation_mask = source_validation_mask.to(device)
    predictor_train_mask = predictor_train_mask.to(device)
    predictor_validation_mask = predictor_validation_mask.to(device)
    fairness_train_mask = fairness_train_mask.to(device)

    neutralizer = FeatureNeutralizer(args.num_features, args.hidden).to(device)
    encoder = build_encoder(args).to(device)
    classifier = MLP_classifier(args).to(device)
    discriminator = MLP_discriminator(args).to(device)

    optimizer_neutralizer = torch.optim.Adam(
        neutralizer.parameters(), lr=args.m_lr, weight_decay=args.m_wd
    )
    optimizer_encoder = torch.optim.Adam(
        encoder.parameters(), lr=args.e_lr, weight_decay=args.e_wd
    )
    optimizer_classifier = torch.optim.Adam(
        classifier.parameters(), lr=args.c_lr, weight_decay=args.c_wd
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=args.d_lr, weight_decay=args.d_wd
    )
    discriminator_criterion = nn.BCELoss()

    # Source-only F3 warm-up.  The target graph and its sensitive attributes are
    # intentionally not used to construct or refine these regression targets.
    neutralizer.train()
    for _ in range(args.predictor_warmup_epochs):
        optimizer_neutralizer.zero_grad()
        prediction = neutralizer(source.x)
        warmup_loss = F.mse_loss(
            prediction[predictor_train_mask],
            heterogeneous_mean[predictor_train_mask],
        )
        warmup_loss.backward()
        optimizer_neutralizer.step()

    final_validation = None

    for epoch in range(args.epochs):
        # Step 1: the discriminator learns to recover the sensitive attribute
        # from a detached source embedding.
        neutralizer.eval()
        encoder.eval()
        discriminator.train()
        set_requires_grad(discriminator, True)
        for _ in range(args.discriminator_steps):
            optimizer_discriminator.zero_grad()
            with torch.no_grad():
                _, source_embedding, _ = forward_model(
                    source, neutralizer, encoder, classifier, args.delta
                )
            sensitive_probability = discriminator(
                source_embedding[fairness_train_mask]
            ).view(-1)
            discriminator_loss = discriminator_criterion(
                sensitive_probability,
                source.sens[fairness_train_mask].float(),
            )
            discriminator_loss.backward()
            optimizer_discriminator.step()

        # Step 2: classifier/encoder/F3 learn the source task while F3 also
        # matches heterogeneous-neighbor features and fools the discriminator.
        neutralizer.train()
        encoder.train()
        classifier.train()
        discriminator.eval()
        set_requires_grad(discriminator, False)
        for _ in range(args.task_steps):
            optimizer_neutralizer.zero_grad()
            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()

            logits, source_embedding, predicted_heterogeneous = forward_model(
                source, neutralizer, encoder, classifier, args.delta
            )
            task_loss = F.binary_cross_entropy_with_logits(
                logits[source_train_mask], source.y[source_train_mask]
            )
            neutralization_loss = F.mse_loss(
                predicted_heterogeneous[predictor_train_mask],
                heterogeneous_mean[predictor_train_mask],
            )
            adversarial_loss = discriminator_criterion(
                discriminator(source_embedding[fairness_train_mask]).view(-1),
                source.sens[fairness_train_mask].float(),
            )
            total_loss = (
                task_loss
                + args.neutral_weight * neutralization_loss
                - args.adv_weight * adversarial_loss
            )
            total_loss.backward()
            optimizer_neutralizer.step()
            optimizer_encoder.step()
            optimizer_classifier.step()
        set_requires_grad(discriminator, True)

        # Source validation is monitoring only. It never selects an epoch and
        # cannot shorten the fixed training schedule.
        neutralizer.eval()
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            source_logits, _, predicted_heterogeneous = forward_model(
                source, neutralizer, encoder, classifier, args.delta
            )
            validation_metrics = evaluate_logits(
                source_logits,
                source.y,
                source.sens,
                source_validation_mask,
            )
            if predictor_validation_mask.any():
                predictor_validation_loss = F.mse_loss(
                    predicted_heterogeneous[predictor_validation_mask],
                    heterogeneous_mean[predictor_validation_mask],
                ).item()
            else:
                predictor_validation_loss = float("nan")

        score = validation_tradeoff(validation_metrics, args.alpha)
        final_validation = validation_metrics
        should_log = (
            epoch == 0
            or epoch + 1 == args.epochs
            or (args.log_every > 0 and (epoch + 1) % args.log_every == 0)
        )
        if should_log:
            print(
                "run={} epoch={:03d}/{:03d} source_val {} score={:.4f} "
                "F3_MSE={:.6f}".format(
                    run_index + 1,
                    epoch + 1,
                    args.epochs,
                    format_metrics(validation_metrics),
                    score,
                    predictor_validation_loss,
                )
            )

    if final_validation is None:
        raise RuntimeError("Source training completed without validation metrics")

    neutralizer.eval()
    encoder.eval()
    classifier.eval()

    # Save the final fixed-epoch state. No target graph has been loaded yet.
    if not args.no_save:
        checkpoint_path = Path(args.checkpoint_dir) / "run_{:02d}.pt".format(
            run_index + 1
        )
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            neutralizer=neutralizer,
            encoder=encoder,
            classifier=classifier,
            discriminator=discriminator,
            args=args,
            source_validation=final_validation,
            preprocessor=preprocessor,
        )

    artifact = TrainedRunArtifact(
        run_index=run_index,
        neutralizer_state=state_dict_on_cpu(neutralizer),
        encoder_state=state_dict_on_cpu(encoder),
        classifier_state=state_dict_on_cpu(classifier),
    )
    print(
        "run={} SOURCE TRAINING COMPLETE epochs={} final_source_val {}".format(
            run_index + 1,
            args.epochs,
            format_metrics(final_validation),
        )
    )
    return artifact


def evaluate_target_only(
    target_cpu: GraphDomain,
    artifact: TrainedRunArtifact,
    args: argparse.Namespace,
) -> Tuple[TransferMetrics, np.ndarray, np.ndarray]:
    """Evaluate frozen weights with target data only and no parameter updates."""

    device = args.device
    target = target_cpu.to(device)
    neutralizer = FeatureNeutralizer(args.num_features, args.hidden).to(device)
    encoder = build_encoder(args).to(device)
    classifier = MLP_classifier(args).to(device)
    neutralizer.load_state_dict(artifact.neutralizer_state)
    encoder.load_state_dict(artifact.encoder_state)
    classifier.load_state_dict(artifact.classifier_state)

    for module in (neutralizer, encoder, classifier):
        module.eval()
        set_requires_grad(module, False)

    target_mask = target.label_mask
    if not target_mask.any():
        raise RuntimeError("Target domain has no labeled nodes to evaluate")
    with torch.inference_mode():
        target_logits, target_embedding, _ = forward_model(
            target, neutralizer, encoder, classifier, args.delta
        )
        target_metrics = evaluate_logits(
            target_logits, target.y, target.sens, target_mask
        )
        target_representations = target_embedding[target_mask].detach().cpu().numpy()
        target_group_labels = encode_visualization_groups(
            target.y, target.sens, target_mask
        )

    print(
        "run={} FINAL target={} {}".format(
            artifact.run_index + 1, target.name, format_metrics(target_metrics)
        )
    )
    return target_metrics, target_representations, target_group_labels


def summarize(target_results: List[TransferMetrics], source: str, target: str) -> None:
    print("\n====== FairSIN source={} -> target={} ======".format(source, target))
    print("Target metrics over {} run(s); lower DP/EO is fairer.".format(len(target_results)))
    result_dicts = [metrics.as_dict() for metrics in target_results]
    for key in (
        "accuracy",
        "auc_roc",
        "f1",
        "demographic_parity",
        "equal_opportunity",
    ):
        values = np.asarray([result[key] for result in result_dicts], dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            print("{}: nan".format(key))
        else:
            print(
                "{}: {:.2f} +/- {:.2f}".format(
                    key, finite_values.mean() * 100.0, finite_values.std() * 100.0
                )
            )


def resolve_dataset_configuration(
    args: argparse.Namespace,
) -> Tuple[DatasetProfile, Path]:
    """Infer a dataset profile and fill schema defaults without data leakage."""

    if args.dataset_family == "auto":
        named_domains = [name for name in (args.source, args.target) if name]
        if not named_domains:
            profile = DATASET_PROFILES["bailA"]
        else:
            candidates = [
                candidate
                for candidate in DATASET_PROFILES.values()
                if all(name.startswith(candidate.domain_prefix) for name in named_domains)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Could not infer one dataset family from source/target: {}. "
                    "Use --dataset-family explicitly.".format(named_domains)
                )
            profile = candidates[0]
    else:
        profile = DATASET_PROFILES[args.dataset_family]

    if args.source is None:
        args.source = profile.default_source
    if args.target is None:
        args.target = profile.default_target
    if not args.source.startswith(profile.domain_prefix):
        raise ValueError(
            "Source {} does not belong to dataset family {}".format(
                args.source, profile.family
            )
        )
    if not args.target.startswith(profile.domain_prefix):
        raise ValueError(
            "Target {} does not belong to dataset family {}".format(
                args.target, profile.family
            )
        )

    args.dataset_family = profile.family
    args.label_column = args.label_column or profile.label_column
    args.sensitive_column = args.sensitive_column or profile.sensitive_column
    args.id_column = args.id_column or profile.id_column
    args.normalization = (
        profile.normalization if args.normalization == "auto" else args.normalization
    )
    args.drop_columns = tuple(
        dict.fromkeys(profile.drop_columns + tuple(args.drop_column))
    )
    if args.checkpoint_dir is None:
        args.checkpoint_dir = "checkpoints/fairsin_{}_to_{}".format(
            args.source, args.target
        )
    data_dir = Path(args.data_dir or profile.data_dir)
    return profile, data_dir


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Source-only FairSIN transfer for BailA, GermanA, Pokec, or Synthetic"
        )
    )
    parser.add_argument(
        "--dataset-family",
        choices=("auto", "bailA", "germanA", "pokec", "syn"),
        default="auto",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Defaults to the profile path under dataset/ after inference",
    )
    parser.add_argument("--source", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--sensitive-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Additional feature column to exclude; may be repeated",
    )
    parser.add_argument(
        "--normalization",
        choices=("auto", "source-minmax", "none"),
        default="auto",
        help=(
            "auto uses source-minmax for BailA/Pokec/Synthetic and raw "
            "GermanA features"
        ),
    )
    parser.add_argument(
        "--include-id-feature",
        action="store_true",
        help="Include the identifier column as a feature (not recommended)",
    )

    parser.add_argument("--encoder", choices=("GCN", "GIN", "SAGE"), default="GCN")
    parser.add_argument("--prop", choices=("scatter", "spmm"), default="scatter")
    parser.add_argument("--hidden", type=int, default=18)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--neutral-weight", type=float, default=1.0)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Fairness penalty in the source-validation monitoring score",
    )

    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=150)
    parser.add_argument("--predictor-warmup-epochs", type=int, default=100)
    parser.add_argument("--task-steps", type=int, default=10)
    parser.add_argument("--discriminator-steps", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--m-lr", type=float, default=0.1)
    parser.add_argument("--m-wd", type=float, default=0.001)
    parser.add_argument("--e-lr", type=float, default=0.1)
    parser.add_argument("--e-wd", type=float, default=0.001)
    parser.add_argument("--c-lr", type=float, default=0.1)
    parser.add_argument("--c-wd", type=float, default=0.001)
    parser.add_argument("--d-lr", type=float, default=0.01)
    parser.add_argument("--d-wd", type=float, default=0.0)
    parser.add_argument("--device", default="auto")

    parser.add_argument(
        "--checkpoint-dir",
        default=None,
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile, data_dir = resolve_dataset_configuration(args)
    args.device = resolve_device(args.device)
    feature_names, alignment_info = resolve_aligned_feature_names(
        data_dir=data_dir,
        source_name=args.source,
        target_name=args.target,
        profile=profile,
        label_column=args.label_column,
        sensitive_column=args.sensitive_column,
        id_column=args.id_column,
        include_id_feature=args.include_id_feature,
        drop_columns=args.drop_columns,
    )

    source = load_graph_domain(
        data_dir=data_dir,
        domain_name=args.source,
        profile=profile,
        label_column=args.label_column,
        sensitive_column=args.sensitive_column,
        id_column=args.id_column,
        feature_names=feature_names,
    )
    source, preprocessor = fit_source_preprocessor(
        source, args.sensitive_column, args.normalization
    )

    args.num_features = source.x.shape[1]
    args.num_classes = 1
    args.sens_idx = preprocessor.sensitive_index
    heterogeneous_mean, has_heterogeneous_neighbor, heterogeneous_degree = (
        heterogeneous_neighbor_targets(source)
    )

    print(
        json.dumps(
            {
                "phase": "source_training",
                "protocol": "fixed epochs/final state/no early stopping",
                "target_loaded": False,
                "target_header_only_inspected": alignment_info.get(
                    "target_header_only_inspected_before_training", False
                ),
                "target_schema_row_only_inspected": alignment_info.get(
                    "target_schema_row_only_inspected_before_training", False
                ),
                "dataset_family": args.dataset_family,
                "source": args.source,
                "label_column": args.label_column,
                "sensitive_column": args.sensitive_column,
                "dropped_feature_columns": args.drop_columns,
                "normalization": args.normalization,
                "source_nodes": source.num_nodes,
                "source_labeled_nodes": int(source.label_mask.sum().item()),
                "source_unlabeled_nodes": int((~source.label_mask).sum().item()),
                "num_features": args.num_features,
                "feature_alignment": alignment_info,
                "identifier_feature_included": args.include_id_feature,
                "source_directed_edges_after_symmetrization": int(source.adj.nnz),
                "source_mean_heterogeneous_degree": float(heterogeneous_degree.mean()),
                "source_nodes_without_heterogeneous_neighbor": int(
                    (~has_heterogeneous_neighbor).sum().item()
                ),
                "runs": args.runs,
                "epochs_per_run": args.epochs,
                "device": str(args.device),
            },
            indent=2,
        )
    )

    trained_artifacts = []
    for run_index in range(args.runs):
        artifact = train_source_once(
            source_cpu=source,
            heterogeneous_mean_cpu=heterogeneous_mean,
            has_heterogeneous_neighbor_cpu=has_heterogeneous_neighbor,
            preprocessor=preprocessor,
            args=args,
            run_index=run_index,
        )
        trained_artifacts.append(artifact)
        gc.collect()
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        "ALL SOURCE TRAINING COMPLETE: {} run(s) x {} epoch(s). "
        "Releasing source-domain tensors before target loading.".format(
            args.runs, args.epochs
        )
    )
    del source
    del heterogeneous_mean
    del has_heterogeneous_neighbor
    del heterogeneous_degree
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()

    # The target dataset is deliberately loaded only after every source run has
    # completed and source node-level tensors have been released.
    target = load_graph_domain(
        data_dir=data_dir,
        domain_name=args.target,
        profile=profile,
        label_column=args.label_column,
        sensitive_column=args.sensitive_column,
        id_column=args.id_column,
        feature_names=feature_names,
    )
    target_outside_source_range = count_values_outside_source_range(
        target, preprocessor
    )
    target = apply_source_preprocessor(target, preprocessor)
    print(
        json.dumps(
            {
                "phase": "target_test",
                "protocol": "target data only/frozen final source weights/inference only",
                "source_graph_available": False,
                "parameter_updates": False,
                "target": args.target,
                "target_nodes": target.num_nodes,
                "target_labeled_nodes_evaluated": int(
                    target.label_mask.sum().item()
                ),
                "target_unlabeled_nodes_excluded": int(
                    (~target.label_mask).sum().item()
                ),
                "target_directed_edges_after_symmetrization": int(target.adj.nnz),
                "target_feature_values_outside_source_observed_range": (
                    target_outside_source_range
                ),
                "device": str(args.device),
            },
            indent=2,
        )
    )

    target_results = []
    for artifact in trained_artifacts:
        (
            target_metrics,
            target_representations,
            target_group_labels,
        ) = evaluate_target_only(
            target_cpu=target,
            artifact=artifact,
            args=args,
        )
        target_results.append(target_metrics)
        # Export only after frozen target inference has completed.  The output
        # pair is written directly to the process current working directory.
        save_visualization_embeddings_current_folder(
            representations=target_representations,
            labels=target_group_labels,
            run_index=artifact.run_index,
        )
        gc.collect()
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    summarize(target_results, args.source, args.target)


if __name__ == "__main__":
    main()
