"""Source-free GALA adaptation for large single-graph fairness datasets.

The public GALA repository is written for TU graph-classification datasets,
where every sample is a separate graph.  bailA and germanA are instead pairs
of large single graphs.  This runner keeps the three GALA ideas intact while
adapting their implementation to a transductive node-classification setting:

1. a source-style sparse graph diffusion/reconstruction step;
2. class-wise, curriculum pseudo-labeling on unlabeled target nodes; and
3. graph-jigsaw consistency regularization.

The dataset label is stored only as the supervised label tensor and the
sensitive attribute only as the evaluation tensor.  Neither column is ever
included in ``x``.  ``user_id`` is used for neither features nor indexing:
edge files are indexed by row position, which is how the supplied files are
encoded.

This file intentionally contains no data-dependent shortcuts or Python-side
execution assumptions; it is meant to be copied to a training server and run
there with the repository's PyTorch/PyG environment.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, to_undirected
from utils.device import resolve_device


DATASET_CONFIGS = {
    "baila": {
        "dataset_name": "bailA",
        "label_column": "RECID",
        "sensitive_column": "WHITE",
        "source_domain": "bailA_2",
        "target_domain": "bailA_1",
        "domain_is_directory": False,
        "edge_id_column": None,
        "label_mode": "binary",
    },
    "germana": {
        "dataset_name": "germanA",
        "label_column": "GoodCustomer",
        "sensitive_column": "Gender",
        "source_domain": "germanA_2",
        "target_domain": "germanA_1",
        "domain_is_directory": False,
        "edge_id_column": None,
        "label_mode": "binary",
    },
    "pokec": {
        "dataset_name": "pokec",
        "label_column": "I_am_working_in_field",
        "sensitive_column": "region",
        "source_domain": "pokec_z",
        "target_domain": "pokec_n",
        "domain_is_directory": True,
        "edge_id_column": "user_id",
        "label_mode": "pokec_working_field",
        "storage_format": "csv_table",
    },
    "syn": {
        "dataset_name": "syn",
        "label_column": "label",
        "sensitive_column": "sensitive",
        "source_domain": "syn-2",
        "target_domain": "syn-1",
        "domain_is_directory": False,
        "edge_id_column": None,
        "label_mode": "binary",
        "storage_format": "split_files",
    },
}


def set_seed(seed: int) -> None:
    """Set all RNGs used by the experiment."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _as_float(value: str) -> float:
    value = value.strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
        return 0.0
    return float(value)


def _is_numeric(value: str) -> bool:
    value = value.strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False


def _read_edges(
    edge_path: Path,
    num_nodes: int,
    node_id_to_index: Dict[int, int] = None,
) -> torch.Tensor:
    """Read and symmetrize the row-indexed edge list.

    The distributed fairness-graph edge files may contain self-loops and,
    depending on the
    source, may contain one or both directions.  GCNConv adds self-loops, so
    we remove them here and explicitly make the graph undirected.
    """

    edges: List[Tuple[int, int]] = []
    skipped = 0
    with edge_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            # Supports whitespace/tab-separated edge files and the comma-
            # separated edge format used by the synthetic datasets.
            parts = line.strip().replace(",", " ").split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"Malformed edge at {edge_path}:{line_number}: {line!r}")
            # bailA uses integer text while germanA stores the same row
            # indices in scientific-notation float text.
            src_raw, dst_raw = int(float(parts[0])), int(float(parts[1]))
            if node_id_to_index is not None:
                # Pokec edges use original user_id values, not CSV row indices.
                if src_raw not in node_id_to_index or dst_raw not in node_id_to_index:
                    skipped += 1
                    continue
                src, dst = node_id_to_index[src_raw], node_id_to_index[dst_raw]
            else:
                src, dst = src_raw, dst_raw
                if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
                    raise ValueError(
                        f"Edge ({src}, {dst}) in {edge_path} is outside "
                        f"the {num_nodes}-node graph."
                    )
            if src != dst:
                edges.append((src, dst))

    if not edges:
        raise ValueError(f"No non-self edges found in {edge_path}.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    if skipped:
        print(f"[data] skipped {skipped} edges with endpoints outside {edge_path.name}")
    return edge_index.contiguous()


def _build_label_mapping(values: Sequence[str]) -> Dict[str, int]:
    normalized = [str(value).strip() for value in values]
    unique = sorted(set(normalized))
    if set(unique) == {"-1", "1"}:
        return {"-1": 0, "1": 1}
    if set(unique) == {"0", "1"}:
        return {"0": 0, "1": 1}
    if len(unique) != 2:
        raise ValueError(f"Expected a binary column, found values: {unique}")
    return {value: index for index, value in enumerate(unique)}


def _encode_labels(
    values: Sequence[str],
    label_mode: str,
    label_mapping: Dict[str, int] = None,
) -> Tuple[List[int], List[bool], Dict[str, object]]:
    normalized = [str(value).strip() for value in values]
    if label_mode == "pokec_working_field":
        # -1 denotes an unavailable working-field label.  It remains in the
        # graph for message passing and target adaptation, but is excluded from
        # source supervision and final target metrics.  0 is the negative
        # class and every positive field category (1, 2, ...) is class 1.
        labels: List[int] = []
        valid: List[bool] = []
        for value in normalized:
            numeric = int(float(value))
            is_valid = numeric >= 0
            valid.append(is_valid)
            labels.append(1 if numeric > 0 else 0)
        mapping: Dict[str, object] = {"-1": "ignored", "0": 0, ">0": 1}
        return labels, valid, mapping

    if label_mode != "binary":
        raise ValueError(f"Unknown label mode: {label_mode}")
    if label_mapping is None:
        label_mapping = _build_label_mapping(normalized)
    labels = [int(label_mapping[value]) for value in normalized]
    return labels, [True] * len(labels), label_mapping


def _build_feature_schema(
    fields: Sequence[str],
    rows: Sequence[Dict[str, str]],
    excluded: set,
    feature_columns: Sequence[str] = None,
) -> Dict[str, object]:
    raw_features = list(feature_columns) if feature_columns is not None else [
        name for name in fields if name not in excluded
    ]
    missing_features = [name for name in raw_features if name not in fields]
    if missing_features:
        raise ValueError(f"Feature columns are missing from CSV: {missing_features}")
    categorical: Dict[str, List[str]] = {}
    for name in raw_features:
        if not all(_is_numeric(row.get(name, "")) for row in rows):
            categorical[name] = sorted({str(row.get(name, "")).strip() for row in rows})

    encoded_names: List[str] = []
    for name in raw_features:
        if name in categorical:
            encoded_names.extend([f"{name}={value}" for value in categorical[name]])
        else:
            encoded_names.append(name)
    return {
        "raw_features": raw_features,
        "categorical": categorical,
        "encoded_names": encoded_names,
    }


def _encode_features(rows: Sequence[Dict[str, str]], schema: Dict[str, object]) -> List[List[float]]:
    raw_features = schema["raw_features"]
    categorical = schema["categorical"]
    features: List[List[float]] = []
    for row in rows:
        encoded: List[float] = []
        for name in raw_features:
            if name in categorical:
                value = str(row.get(name, "")).strip()
                encoded.extend([1.0 if value == category else 0.0 for category in categorical[name]])
            else:
                encoded.append(_as_float(row.get(name, "")))
        features.append(encoded)
    return features


def _domain_paths(
    data_root: str,
    dataset_name: str,
    domain: str,
    domain_is_directory: bool,
) -> Tuple[Path, Path]:
    root = Path(data_root) / (domain if domain_is_directory else dataset_name)
    return root / f"{domain}.csv", root / f"{domain}_edges.txt"


def _read_csv_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty CSV: {csv_path}") from exc


def _normalize_domain_name(dataset_key: str, domain: str) -> str:
    """Normalize user-facing aliases such as syn1 to on-disk syn-1 names."""

    value = str(domain).strip()
    if dataset_key != "syn":
        return value
    compact = value.lower().replace("_", "").replace("-", "")
    aliases = {"syn1": "syn-1", "syn2": "syn-2"}
    if compact not in aliases:
        raise ValueError(
            f"Unknown synthetic domain {domain!r}. Use syn-1/syn1 or syn-2/syn2."
        )
    return aliases[compact]


def load_syn_graph(data_root: str, domain: str) -> Data:
    """Load a synthetic domain stored in four separate files."""

    root = Path(data_root) / "syn"
    feature_path = root / f"{domain}_feat.csv"
    label_path = root / f"{domain}_label.txt"
    sensitive_path = root / f"{domain}_sens.txt"
    edge_path = root / f"{domain}_edges.txt"
    required_paths = [feature_path, label_path, sensitive_path, edge_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing synthetic data files: {missing}")

    features: List[List[float]] = []
    with feature_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            try:
                features.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric feature at {feature_path}:{line_number}"
                ) from exc
    if not features:
        raise ValueError(f"No feature rows found in {feature_path}")
    feature_dim = len(features[0])
    if any(len(row) != feature_dim for row in features):
        raise ValueError(f"Inconsistent feature dimensions in {feature_path}")

    def read_binary_vector(path: Path, name: str) -> List[int]:
        values: List[int] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = line.strip()
                if value == "":
                    continue
                numeric = int(float(value))
                if numeric not in {0, 1}:
                    raise ValueError(
                        f"{name} must be binary at {path}:{line_number}; found {value!r}"
                    )
                values.append(numeric)
        return values

    labels = read_binary_vector(label_path, "label")
    sensitive = read_binary_vector(sensitive_path, "sensitive attribute")
    num_nodes = len(features)
    if len(labels) != num_nodes or len(sensitive) != num_nodes:
        raise ValueError(
            f"Synthetic domain {domain} has mismatched lengths: "
            f"features={num_nodes}, labels={len(labels)}, sensitive={len(sensitive)}"
        )

    feature_names = [f"feature_{index}" for index in range(feature_dim)]
    feature_schema = {
        "raw_features": feature_names,
        "categorical": {},
        "encoded_names": feature_names,
    }
    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        label_valid=torch.ones(num_nodes, dtype=torch.bool),
        sensitive=torch.tensor(sensitive, dtype=torch.long),
        node_id=torch.arange(num_nodes, dtype=torch.long),
    )
    data.edge_index = _read_edges(edge_path, num_nodes)
    data.feature_names = feature_names
    data.raw_feature_names = feature_names
    data.feature_schema = feature_schema
    data.label_column = "label"
    data.sensitive_column = "sensitive"
    data.label_mode = "binary"
    data.label_mapping = {"0": 0, "1": 1}
    data.sensitive_mapping = {"0": 0, "1": 1}
    data.dataset_name = "syn"
    data.domain = domain
    return data


def load_fair_graph(
    data_root: str,
    dataset_name: str,
    domain: str,
    label_column: str,
    sensitive_column: str,
    domain_is_directory: bool = False,
    edge_id_column: str = None,
    label_mode: str = "binary",
    feature_columns: Sequence[str] = None,
    feature_schema: Dict[str, object] = None,
    label_mapping: Dict[str, object] = None,
    sensitive_mapping: Dict[str, int] = None,
) -> Data:
    """Load one fairness domain without leaking IDs or sensitive features.

    Categorical input columns (notably germanA's ``PurposeOfLoan``) are
    one-hot encoded.  The schema and mappings are fitted on the source domain
    and reused for the target domain, so target statistics do not influence
    preprocessing.
    """

    csv_path, edge_path = _domain_paths(
        data_root, dataset_name, domain, domain_is_directory
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {dataset_name} CSV: {csv_path}")
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing {dataset_name} edge list: {edge_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")
        fields = list(reader.fieldnames)
        required = {"user_id", label_column, sensitive_column}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        rows = list(reader)
        excluded = {"user_id", label_column, sensitive_column}
        if feature_schema is None:
            feature_schema = _build_feature_schema(
                fields, rows, excluded, feature_columns=feature_columns
            )
        missing_schema_features = [
            name for name in feature_schema["raw_features"] if name not in fields
        ]
        if missing_schema_features:
            raise ValueError(
                f"Source feature columns are absent from {csv_path}: {missing_schema_features}"
            )

        labels, label_valid, label_mapping = _encode_labels(
            [row[label_column] for row in rows], label_mode, label_mapping
        )
        if sensitive_mapping is None:
            sensitive_mapping = _build_label_mapping([row[sensitive_column] for row in rows])

        features = _encode_features(rows, feature_schema)
        sensitive = [sensitive_mapping[str(row[sensitive_column]).strip()] for row in rows]

    if not features:
        raise ValueError(f"No rows found in {csv_path}")
    if not set(labels).issubset({0, 1}):
        raise ValueError(f"{label_column} must be binary in {csv_path}; found {sorted(set(labels))}")
    if not set(sensitive).issubset({0, 1}):
        raise ValueError(f"{sensitive_column} must be binary in {csv_path}; found {sorted(set(sensitive))}")

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        label_valid=torch.tensor(label_valid, dtype=torch.bool),
        sensitive=torch.tensor(sensitive, dtype=torch.long),
        node_id=torch.arange(len(features), dtype=torch.long),
    )
    node_id_to_index = None
    if edge_id_column is not None:
        if edge_id_column not in fields:
            raise ValueError(f"Missing edge ID column {edge_id_column!r} in {csv_path}")
        original_node_ids = [int(float(row[edge_id_column])) for row in rows]
        if len(set(original_node_ids)) != len(original_node_ids):
            raise ValueError(f"Duplicate {edge_id_column} values in {csv_path}")
        node_id_to_index = {
            original_id: index for index, original_id in enumerate(original_node_ids)
        }
        data.original_node_id = torch.tensor(original_node_ids, dtype=torch.long)
    data.edge_index = _read_edges(edge_path, data.num_nodes, node_id_to_index)
    data.feature_names = feature_schema["encoded_names"]
    data.raw_feature_names = feature_schema["raw_features"]
    data.feature_schema = feature_schema
    data.label_column = label_column
    data.sensitive_column = sensitive_column
    data.label_mode = label_mode
    data.label_mapping = label_mapping
    data.sensitive_mapping = sensitive_mapping
    data.dataset_name = dataset_name
    data.domain = domain
    return data


def standardize_features(source: Data, target: Data) -> Tuple[Data, Data]:
    """Standardize using source statistics only.

    The target graph is never used to estimate preprocessing statistics, which
    keeps this step source-free in the strict adaptation sense.
    """

    source = copy.deepcopy(source)
    target = copy.deepcopy(target)
    mean = source.x.mean(dim=0, keepdim=True)
    std = source.x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    source.x = (source.x - mean) / std
    target.x = (target.x - mean) / std
    source.feature_mean = mean.squeeze(0)
    source.feature_std = std.squeeze(0)
    target.feature_mean = mean.squeeze(0)
    target.feature_std = std.squeeze(0)
    return source, target


def stratified_mask(
    labels: torch.Tensor,
    train_ratio: float,
    seed: int,
    eligible: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic, per-class train/held-out masks."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train/adaptation ratio must be between 0 and 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    if eligible is None:
        eligible = torch.ones(labels.numel(), dtype=torch.bool)
    eligible = eligible.bool()
    for cls in torch.unique(labels[eligible], sorted=True).tolist():
        idx = torch.where(eligible & labels.eq(int(cls)))[0]
        if idx.numel() < 2:
            train_mask[idx] = True
            continue
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        count = int(round(idx.numel() * train_ratio))
        count = min(max(count, 1), idx.numel() - 1)
        train_mask[perm[:count]] = True
    return train_mask, eligible & ~train_mask


def random_mask(num_nodes: int, train_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split target nodes without reading target labels.

    Target labels are reserved for the final metrics.  In
    particular, the adaptation/test split itself must not be stratified by
    target labels because that would leak the evaluation label distribution.
    """

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train/adaptation ratio must be between 0 and 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_nodes, generator=generator)
    count = min(max(int(round(num_nodes * train_ratio)), 1), num_nodes - 1)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[permutation[:count]] = True
    return train_mask, ~train_mask


class GCNEncoder(nn.Module):
    """The GALA encoder, fixed to the paper's GCN choice."""

    def __init__(self, in_channels: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("num_gc_layers must be at least 1")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            self.convs.append(GCNConv(in_channels if layer == 0 else hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.norms[layer](h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GCNNodeClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, layers: int, dropout: float, num_classes: int):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_dim, layers, dropout)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x, edge_index)
        projected = F.normalize(self.proj_head(embedding), dim=-1)
        return self.classifier(embedding), projected


class SparseGraphDiffusionAligner:
    """Source-style graph reconstruction for a large sparse single graph.

    The original GALA diffusion model operates on dense adjacency matrices of
    graph samples.  A fairness domain is one potentially large graph, so a
    dense adjacency tensor is infeasible.  This implementation is the sparse
    equivalent: source edge-density statistics define the learned source
    style, and a forward-noise/reverse-reconstruction schedule stochastically
    removes or adds edges until the target graph approaches that style.  It
    preserves node features and labels and never uses target labels.
    """

    def __init__(self, source_edge_index: torch.Tensor, source_nodes: int, steps: int, noise: float):
        if steps < 1:
            raise ValueError("diffusion_steps must be at least 1")
        self.steps = int(steps)
        self.noise = float(noise)
        source_pairs = self._unique_undirected_pairs(source_edge_index)
        self.source_mean_degree = 2.0 * len(source_pairs) / max(1, source_nodes)

    @staticmethod
    def _unique_undirected_pairs(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()
        return sorted({(min(a, b), max(a, b)) for a, b in zip(src, dst) if a != b})

    @staticmethod
    def _to_edge_index(pairs: Iterable[Tuple[int, int]]) -> torch.Tensor:
        pair_list = list(pairs)
        if not pair_list:
            raise ValueError("Diffusion reconstruction produced an edgeless graph")
        directed = pair_list + [(b, a) for a, b in pair_list]
        return torch.tensor(directed, dtype=torch.long).t().contiguous()

    def reconstruct(self, target_edge_index: torch.Tensor, target_nodes: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        pairs = self._unique_undirected_pairs(target_edge_index)
        if not pairs:
            return target_edge_index

        desired = max(1, int(round(self.source_mean_degree * target_nodes / 2.0)))
        # The schedule is analogous to the forward SDE followed by reverse
        # denoising: each step moves only a fraction of the remaining density
        # gap, avoiding a one-shot destructive rewrite of the target graph.
        for step in range(self.steps):
            progress = (step + 1) / self.steps
            stochastic = 1.0 + self.noise * (torch.rand((), generator=generator).item() - 0.5)
            target_count = int(round(len(pairs) + (desired - len(pairs)) * progress * stochastic))
            target_count = max(1, target_count)

            if target_count < len(pairs):
                order = torch.randperm(len(pairs), generator=generator)[:target_count].tolist()
                pairs = [pairs[index] for index in order]
                continue

            if target_count == len(pairs):
                continue

            # Add random non-edges only when the source style is denser.  The
            # cap prevents pathological quadratic work on unusually sparse
            # graphs while still matching the source-domain degree.
            needed = min(target_count - len(pairs), max(10000, len(pairs)))
            pair_set = set(pairs)
            attempts = 0
            max_attempts = max(needed * 20, 100)
            while needed > 0 and attempts < max_attempts:
                attempts += 1
                a = int(torch.randint(target_nodes, (1,), generator=generator).item())
                b = int(torch.randint(target_nodes, (1,), generator=generator).item())
                if a == b:
                    continue
                pair = (min(a, b), max(a, b))
                if pair in pair_set:
                    continue
                pair_set.add(pair)
                pairs.append(pair)
                needed -= 1

        return self._to_edge_index(pairs)


def graph_jigsaw_view(
    data: Data, ratio: float, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a feature-preserving node-level graph-jigsaw view.

    For a single large graph, subgraphs are represented by randomly sampled
    node blocks.  Swapping the feature blocks is the node analogue of the
    original graph-jigsaw subgraph exchange; the model is constrained to keep
    predictions consistent on the original and exchanged views.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = data.x.clone()
    number = max(2, int(round(data.num_nodes * ratio)))
    number = min(number, data.num_nodes)
    nodes = torch.randperm(data.num_nodes, generator=generator)[:number]
    other = nodes.roll(1)
    original = x[nodes].clone()
    x[nodes] = x[other]
    x[other] = original

    edge_index = data.edge_index
    if edge_index.size(1) > 2:
        keep_probability = max(0.0, 1.0 - min(0.25, ratio))
        keep = torch.rand(edge_index.size(1), generator=generator) < keep_probability
        # Keep at least one edge and keep the graph usable by GCNConv.
        if int(keep.sum()) < 2:
            keep[:2] = True
        edge_index = edge_index[:, keep]
    return x, edge_index


def _class_weights(labels: torch.Tensor, mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels[mask], minlength=num_classes).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (counts[present] * present.sum())
    weights[~present] = 0.0
    return weights


def _pseudo_labels(
    teacher_logits: torch.Tensor,
    adapt_mask: torch.Tensor,
    epoch: int,
    total_epochs: int,
    initial_threshold: float,
    minimum_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class-wise confidence thresholds with a curriculum schedule."""

    probabilities = torch.softmax(teacher_logits.detach(), dim=-1)
    confidence, labels = probabilities.max(dim=-1)
    progress = min(1.0, epoch / max(1, total_epochs))
    scheduled = initial_threshold + (minimum_threshold - initial_threshold) * progress
    thresholds = torch.full(
        (probabilities.size(1),), float(scheduled), device=probabilities.device
    )
    selected = torch.zeros_like(adapt_mask)

    # Each class gets its own empirical confidence threshold.  The lower
    # quartile is used as the curriculum's class-specific floor, then clipped
    # by the global schedule so pseudo-labels remain clean early on.
    for cls in range(probabilities.size(1)):
        class_nodes = adapt_mask & labels.eq(cls)
        if int(class_nodes.sum()) == 0:
            continue
        empirical = torch.quantile(confidence[class_nodes], 0.25).item()
        thresholds[cls] = max(float(scheduled), min(0.99, empirical))
        selected |= class_nodes & confidence.ge(thresholds[cls])
    return labels, selected, thresholds


def _roc_auc(y_true: torch.Tensor, scores: torch.Tensor) -> float:
    """Binary ROC-AUC without adding a runtime dependency on sklearn."""

    y = y_true.detach().cpu().numpy().astype(np.int64)
    s = scores.detach().cpu().numpy().astype(np.float64)
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum = original_ranks[y == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def fairness_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
    sensitive: torch.Tensor,
) -> Dict[str, float]:
    """Return the reference class-wise accuracy/AUC and fairness gaps.

    ``accuracy`` and ``AUC_ROC`` are macro-averaged over target classes, as in
    ``learn(1).py``. ``dp`` is the absolute positive-prediction-rate gap and
    ``eo`` is the absolute true-positive-rate gap (equality of opportunity).
    Every returned value uses percentage points. Lower ``dp``/``eo`` values
    are fairer. The sensitive attribute is used only here, after inference.
    """

    labels = labels.detach().cpu().long()
    predictions = predictions.detach().cpu().long()
    probabilities = probabilities.detach().cpu()
    sensitive = sensitive.detach().cpu().long()

    class_accuracies: List[float] = []
    class_auc_rocs: List[float] = []
    positive_probabilities = probabilities[:, 1]
    for target_class in torch.unique(labels).tolist():
        target_class = int(target_class)
        class_mask = labels.eq(target_class)
        class_accuracies.append(
            float((predictions[class_mask] == labels[class_mask]).float().mean().item())
        )
        one_vs_rest_labels = labels.eq(target_class).long()
        class_scores = (
            positive_probabilities if target_class == 1 else 1.0 - positive_probabilities
        )
        class_auc_rocs.append(
            _roc_auc(one_vs_rest_labels, class_scores)
        )

    def percentage_nanmean(values: Sequence[float]) -> float:
        finite = np.asarray(
            [value for value in values if np.isfinite(value)], dtype=np.float64
        )
        return float(finite.mean() * 100.0) if finite.size else float("nan")

    metrics: Dict[str, float] = {
        "accuracy": percentage_nanmean(class_accuracies),
        "AUC_ROC": percentage_nanmean(class_auc_rocs),
    }

    groups = [sensitive == value for value in (0, 1)]
    positive_rates: List[float] = []
    true_positive_rates: List[float] = []
    for group in groups:
        if int(group.sum()) == 0:
            positive_rates.append(float("nan"))
        else:
            positive_rates.append(float(predictions[group].float().mean().item()))
        positives = group & labels.eq(1)
        if int(positives.sum()) == 0:
            true_positive_rates.append(float("nan"))
        else:
            true_positive_rates.append(float(predictions[positives].float().mean().item()))

    metrics["dp"] = float(abs(positive_rates[0] - positive_rates[1]) * 100.0)
    metrics["eo"] = float(abs(true_positive_rates[0] - true_positive_rates[1]) * 100.0)
    return metrics


def _evaluate(model: GCNNodeClassifier, data: Data, mask: torch.Tensor, device: torch.device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x.to(device), data.edge_index.to(device))
        probabilities = torch.softmax(logits[mask.to(device)], dim=-1)
        predictions = probabilities.argmax(dim=-1)
    return fairness_metrics(
        data.y[mask], predictions.cpu(), probabilities.cpu(), data.sensitive[mask]
    )


def _encode_prediction_sensitive_groups(
    predictions: np.ndarray,
    sensitive: np.ndarray,
) -> np.ndarray:
    """Encode binary (predicted Y, S) pairs using the visualization convention.

    The zyt/visualization reference uses the following fixed group order:
    0=(Y=1,S=0), 1=(Y=1,S=1), 2=(Y=0,S=0), 3=(Y=0,S=1).
    Here Y is deliberately the final target prediction requested for export,
    rather than the target ground-truth label.
    """

    prediction_values = np.asarray(predictions).astype(np.int64, copy=False).reshape(-1)
    sensitive_values = np.asarray(sensitive).astype(np.int64, copy=False).reshape(-1)
    if prediction_values.shape[0] != sensitive_values.shape[0]:
        raise ValueError(
            "Target predictions and sensitive attributes must have the same length: "
            f"{prediction_values.shape[0]} vs {sensitive_values.shape[0]}"
        )
    if not np.isin(prediction_values, [0, 1]).all():
        raise ValueError("Exported target predictions must contain only binary values 0 and 1")
    if not np.isin(sensitive_values, [0, 1]).all():
        raise ValueError("Exported target sensitive attributes must contain only 0 and 1")

    labels = np.full(prediction_values.shape[0], -1, dtype=np.int64)
    labels[(prediction_values == 1) & (sensitive_values == 0)] = 0
    labels[(prediction_values == 1) & (sensitive_values == 1)] = 1
    labels[(prediction_values == 0) & (sensitive_values == 0)] = 2
    labels[(prediction_values == 0) & (sensitive_values == 1)] = 3
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError("Joint visualization labels must contain only 0, 1, 2, or 3")
    return labels


def _export_target_visualization(
    model: GCNNodeClassifier,
    data: Data,
    valid_mask: torch.Tensor,
    device: torch.device,
    output_root: Path,
    seed: int,
) -> Tuple[Path, Path]:
    """Export final target GCN features and predicted-Y/sensitive groups."""

    valid_mask = valid_mask.detach().cpu().bool().reshape(-1)
    if valid_mask.numel() != data.num_nodes:
        raise ValueError(
            f"Target export mask length mismatch: {valid_mask.numel()} vs {data.num_nodes}"
        )
    if int(valid_mask.sum()) == 0:
        raise ValueError("No valid target nodes are available for visualization export")

    model.eval()
    valid_mask_device = valid_mask.to(device)
    with torch.no_grad():
        # The reference visualization exports the final hidden representation
        # immediately before the classifier.  Compute it on the complete target
        # graph first, then apply exactly the same all-valid-node mask to every
        # exported array.
        embeddings = model.encoder(data.x.to(device), data.edge_index.to(device))
        logits = model.classifier(embeddings)
        predictions = logits.argmax(dim=-1)

    representations = (
        embeddings[valid_mask_device].detach().cpu().numpy().astype(np.float32, copy=False)
    )
    predicted_y = predictions[valid_mask_device].detach().cpu().numpy()
    sensitive = data.sensitive[valid_mask].detach().cpu().numpy()
    labels = _encode_prediction_sensitive_groups(predicted_y, sensitive)

    if representations.ndim != 2:
        raise ValueError(
            "representations must be a 2D array shaped [num_valid_target_all_nodes, feature_dim]"
        )
    if representations.shape[0] != labels.shape[0]:
        raise ValueError(
            "representations and labels length mismatch: "
            f"{representations.shape[0]} vs {labels.shape[0]}"
        )
    if not np.isfinite(representations).all():
        raise ValueError("representations contains NaN or Inf; refusing to write invalid NPZ files")
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError("labels contains values outside the allowed set {0, 1, 2, 3}")

    # Five seeds must not overwrite one another.  Each seed therefore gets a
    # directory inside the current experiment output directory, while the two
    # filenames and NPZ keys remain exactly those required by visualization.
    seed_output_dir = (Path(output_root) / f"seed_{seed}").resolve()
    current_directory = Path.cwd().resolve()
    try:
        seed_output_dir.relative_to(current_directory)
    except ValueError as exc:
        raise ValueError(
            "Visualization NPZ files must stay inside the current working directory; "
            f"refusing output path {seed_output_dir}"
        ) from exc
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    feat_path = seed_output_dir / "feat.npz"
    labels_path = seed_output_dir / "labels.npz"
    np.savez_compressed(feat_path, representations=representations)
    np.savez_compressed(labels_path, labels=labels)
    print(
        f"[seed {seed}] exported {feat_path} key=representations shape={representations.shape}; "
        f"{labels_path} key=labels shape={labels.shape}"
    )
    return feat_path, labels_path


def _train_source(
    model: GCNNodeClassifier,
    source: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
    args,
) -> GCNNodeClassifier:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    weights = _class_weights(source.y, train_mask, 2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -float("inf")
    stale = 0
    source_x = source.x.to(device)
    source_edges = source.edge_index.to(device)
    source_y = source.y.to(device)

    for epoch in range(1, args.source_epochs + 1):
        model.train()
        logits, _ = model(source_x, source_edges)
        loss = criterion(logits[train_mask.to(device)], source_y[train_mask.to(device)])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits, _ = model(source_x, source_edges)
            validation_accuracy = float(
                (validation_logits[val_mask.to(device)].argmax(dim=-1)
                 == source_y[val_mask.to(device)]).float().mean().item()
            )
        if validation_accuracy > best_val:
            best_val = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    return model


def _adapt_target(
    model: GCNNodeClassifier,
    target: Data,
    adapt_mask: torch.Tensor,
    aligned_edges: torch.Tensor,
    device: torch.device,
    args,
    seed: int,
) -> GCNNodeClassifier:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    target_x = target.x.to(device)
    target_edges = aligned_edges.to(device)
    aligned_target = copy.deepcopy(target)
    aligned_target.edge_index = aligned_edges
    adapt_mask_device = adapt_mask.to(device)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    for epoch in range(1, args.tta_epoch + 1):
        model.train()
        logits, _ = model(target_x, target_edges)
        with torch.no_grad():
            teacher_logits, _ = teacher(target_x, target_edges)
            pseudo, selected, thresholds = _pseudo_labels(
                teacher_logits,
                adapt_mask_device,
                epoch,
                args.tta_epoch,
                args.pseudo_threshold,
                args.min_pseudo_threshold,
            )

        if int(selected.sum()) > 0:
            pseudo_loss = F.cross_entropy(logits[selected], pseudo[selected])
        else:
            pseudo_loss = logits.sum() * 0.0

        jigsaw_x, jigsaw_edges = graph_jigsaw_view(
            aligned_target, args.jigsaw_ratio, seed * 10000 + epoch
        )
        jigsaw_logits, _ = model(jigsaw_x.to(device), jigsaw_edges.to(device))
        # KL is computed in both directions for the consistency part of GALA.
        student_log_prob = F.log_softmax(logits[adapt_mask_device], dim=-1)
        jigsaw_prob = F.softmax(jigsaw_logits[adapt_mask_device].detach(), dim=-1)
        reverse_log_prob = F.log_softmax(jigsaw_logits[adapt_mask_device], dim=-1)
        student_prob = F.softmax(logits[adapt_mask_device].detach(), dim=-1)
        consistency_loss = 0.5 * (
            F.kl_div(student_log_prob, jigsaw_prob, reduction="batchmean")
            + F.kl_div(reverse_log_prob, student_prob, reduction="batchmean")
        )

        loss = args.pseudo_weight * pseudo_loss + args.consistency_weight * consistency_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # EMA teacher, matching the stable teacher/student implementation used
        # by the original GALA adaptation procedure.
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(teacher.parameters(), model.parameters()):
                teacher_parameter.mul_(args.ema_decay).add_(
                    student_parameter, alpha=1.0 - args.ema_decay
                )

        if epoch == 1 or epoch == args.tta_epoch or epoch % max(1, args.log_interval) == 0:
            print(
                f"[adapt {epoch:03d}/{args.tta_epoch}] loss={loss.item():.6f} "
                f"pseudo={int(selected.sum())}/{int(adapt_mask.sum())} "
                f"thresholds={','.join(f'{v:.3f}' for v in thresholds.detach().cpu().tolist())}"
            )
    return model


def run_one_seed(
    source: Data,
    target: Data,
    aligner: SparseGraphDiffusionAligner,
    args,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, float]:
    set_seed(seed)
    source_train, source_val = stratified_mask(
        source.y, 0.8, seed, eligible=source.label_valid
    )
    # No target labels are consulted for the adaptation/test split.
    target_adapt, target_test = random_mask(target.num_nodes, args.adapt_ratio, seed + 1000)

    model = GCNNodeClassifier(
        source.num_node_features,
        args.hidden_dim,
        args.num_gc_layers,
        args.dropout,
        2,
    ).to(device)
    model = _train_source(model, source, source_train, source_val, device, args)
    aligned_edges = aligner.reconstruct(target.edge_index, target.num_nodes, seed + 2000)
    model = _adapt_target(model, target, target_adapt, aligned_edges, device, args, seed)

    target_eval = copy.deepcopy(target)
    target_eval.edge_index = aligned_edges
    target_metric_mask = target_test & target.label_valid
    if int(target_metric_mask.sum()) == 0:
        raise ValueError("The target test split contains no valid labels")
    metrics = _evaluate(model, target_eval, target_metric_mask, device)
    print(
        f"[seed {seed}] accuracy={metrics['accuracy']:.2f}% "
        f"AUC_ROC={metrics['AUC_ROC']:.2f}% "
        f"dp={metrics['dp']:.2f}% "
        f"eo={metrics['eo']:.2f}%"
    )
    target_all_valid_mask = (target_adapt | target_test) & target.label_valid
    _export_target_visualization(
        model,
        target_eval,
        target_all_valid_mask,
        device,
        output_dir,
        seed,
    )
    return metrics


def _mean_variance(values: Sequence[float]) -> Tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan")
    # Population variance is reported because the requested five seeds are
    # the complete experiment set, rather than a sample from a larger pool.
    return float(finite.mean()), float(finite.var(ddof=0))


def run(args) -> Dict[str, Dict[str, float]]:
    if str(args.conv_type).upper() != "GCN":
        raise ValueError("The fairness GALA experiment is fixed to encoder=GCN.")
    dataset_key = str(args.DS).lower()
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported fairness dataset {args.DS!r}. "
            f"Use one of: {sorted(config['dataset_name'] for config in DATASET_CONFIGS.values())}."
        )
    dataset_config = DATASET_CONFIGS[dataset_key]
    dataset_name = dataset_config["dataset_name"]
    label_column = dataset_config["label_column"]
    sensitive_column = dataset_config["sensitive_column"]
    domain_is_directory = dataset_config["domain_is_directory"]
    edge_id_column = dataset_config["edge_id_column"]
    label_mode = dataset_config["label_mode"]
    storage_format = dataset_config.get("storage_format", "csv_table")
    source_domain = _normalize_domain_name(
        dataset_key, args.source_domain or dataset_config["source_domain"]
    )
    target_domain = _normalize_domain_name(
        dataset_key, args.target_domain or dataset_config["target_domain"]
    )
    output_dir_value = args.output_dir or f"results/{dataset_name}_GALA"
    if (
        source_domain != dataset_config["source_domain"]
        or target_domain != dataset_config["target_domain"]
    ):
        print(f"[config] source={source_domain}, target={target_domain}")

    source_only: List[str] = []
    target_only: List[str] = []
    if storage_format == "split_files":
        source = load_syn_graph(args.data_root, source_domain)
        target = load_syn_graph(args.data_root, target_domain)
        common_features = source.feature_names
    elif storage_format == "csv_table":
        source_csv, _ = _domain_paths(
            args.data_root, dataset_name, source_domain, domain_is_directory
        )
        target_csv, _ = _domain_paths(
            args.data_root, dataset_name, target_domain, domain_is_directory
        )
        source_fields = _read_csv_header(source_csv)
        target_fields = _read_csv_header(target_csv)
        excluded = {"user_id", label_column, sensitive_column}
        common_features = [
            name for name in source_fields if name in target_fields and name not in excluded
        ]
        if not common_features:
            raise ValueError("Source and target domains have no common usable features")
        source_only = [
            name for name in source_fields if name not in target_fields and name not in excluded
        ]
        target_only = [
            name for name in target_fields if name not in source_fields and name not in excluded
        ]
        if source_only or target_only:
            print(
                f"[data] using {len(common_features)} common features; "
                f"dropping source-only={len(source_only)}, target-only={len(target_only)}"
            )

        source = load_fair_graph(
            args.data_root,
            dataset_name,
            source_domain,
            label_column,
            sensitive_column,
            domain_is_directory=domain_is_directory,
            edge_id_column=edge_id_column,
            label_mode=label_mode,
            feature_columns=common_features,
        )
        target = load_fair_graph(
            args.data_root,
            dataset_name,
            target_domain,
            label_column,
            sensitive_column,
            domain_is_directory=domain_is_directory,
            edge_id_column=edge_id_column,
            label_mode=label_mode,
            feature_schema=source.feature_schema,
            label_mapping=source.label_mapping,
            sensitive_mapping=source.sensitive_mapping,
        )
    else:
        raise ValueError(f"Unsupported storage format: {storage_format}")
    if source.x.size(1) != target.x.size(1):
        raise ValueError("Source and target feature dimensions do not match")
    if source.feature_names != target.feature_names:
        raise ValueError("Source and target feature columns do not match in the same order")
    source, target = standardize_features(source, target)
    excluded_display = (
        ["separate label file", "separate sensitive file"]
        if storage_format == "split_files"
        else ["user_id", label_column, sensitive_column]
    )
    print(
        f"[data] source={source_domain} nodes={source.num_nodes} edges={source.edge_index.size(1)}; "
        f"target={target_domain} nodes={target.num_nodes} edges={target.edge_index.size(1)}; "
        f"features={source.num_node_features} "
        f"source_valid_labels={int(source.label_valid.sum())} "
        f"target_valid_labels={int(target.label_valid.sum())}; "
        f"excluded={excluded_display}"
    )

    aligner = SparseGraphDiffusionAligner(
        source.edge_index, source.num_nodes, args.diffusion_steps, args.diffusion_noise
    )
    device = resolve_device(args.device, args.gpu_id)
    print(f"[config] device={device}")
    seeds = [int(value.strip()) for value in str(args.seeds).split(",") if value.strip()]
    if seeds != [1, 2, 3, 4, 5]:
        print(f"[config] seeds={seeds}; requested default is 1,2,3,4,5")
    if not seeds:
        raise ValueError("At least one seed is required")

    output_dir = Path(output_dir_value).resolve()
    current_directory = Path.cwd().resolve()
    try:
        output_dir.relative_to(current_directory)
    except ValueError as exc:
        raise ValueError(
            "The experiment output directory must stay inside the current working directory; "
            f"refusing output path {output_dir}"
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, float]] = []
    for seed in seeds:
        metrics = run_one_seed(source, target, aligner, args, seed, device, output_dir)
        records.append({"seed": seed, **metrics})

    metric_names = ("accuracy", "AUC_ROC", "dp", "eo")
    summary: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        mean, variance = _mean_variance([record[metric] for record in records])
        std = math.sqrt(variance) if np.isfinite(variance) else float("nan")
        summary[metric] = {
            "mean": mean,
            "variance": variance,
            "std": std,
            "formatted": f"{mean:.2f}% +/- {std:.2f}%",
        }

    print(f"\n===== GALA {dataset_name} summary: {source_domain} -> {target_domain} =====")
    for metric in metric_names:
        print(f"{metric}: {summary[metric]['formatted']}")

    output_payload = {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "dataset": dataset_name,
        "encoder": "GCN",
        "device": str(device),
        "feature_columns_excluded": excluded_display,
        "label_column": label_column,
        "sensitive_column": sensitive_column,
        "label_mode": label_mode,
        "common_feature_count": len(common_features),
        "source_valid_label_count": int(source.label_valid.sum()),
        "target_valid_label_count": int(target.label_valid.sum()),
        "source_only_features_dropped": source_only,
        "target_only_features_dropped": target_only,
        "seeds": seeds,
        "per_seed": records,
        "summary": summary,
    }
    result_stem = f"{source_domain}_to_{target_domain}"
    with (output_dir / f"{result_stem}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(output_payload, stream, indent=2, allow_nan=True)
    with (output_dir / f"{result_stem}_per_seed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", *metric_names])
        writer.writeheader()
        writer.writerows(records)
    return summary


def main(args=None) -> Dict[str, Dict[str, float]]:
    if args is None:
        from arguments import arg_parse

        args = arg_parse()
    return run(args)


if __name__ == "__main__":
    main()
