"""Strict graph-data utilities for the GCN-NRC experiments.

Dataset-specific identifiers, labels, sensitive attributes, and unsupported
categorical fields are excluded from model features. Source labels and final
target evaluation columns are exposed by separate functions so target
adaptation cannot accidentally use labels or sensitive attributes.
"""

from __future__ import print_function

import csv
import math

import torch


EXCLUDED_FEATURE_COLUMNS = ("user_id", "RECID", "WHITE")

DATASET_CONFIGURATIONS = {
    "bailA": {
        "dataset_family": "bailA",
        "source_domain": "bailA_2",
        "target_domain": "bailA_1",
        "label_column": "RECID",
        "sensitive_column": "WHITE",
        "excluded_feature_columns": EXCLUDED_FEATURE_COLUMNS,
        "label_mapping": {"0": 0, "1": 1},
        "sensitive_mapping": {"0": 0, "1": 1},
        "normalization": "source_train_standardize",
        "node_id_column": None,
        "target_feature_alignment": "exact",
        "unlabeled_label_value": None,
        "feature_file_has_header": True,
        "source_label_storage": "csv_column",
        "target_evaluation_storage": "csv_columns",
    },
    "germanA": {
        "dataset_family": "germanA",
        "source_domain": "germanA_2",
        "target_domain": "germanA_1",
        "label_column": "GoodCustomer",
        "sensitive_column": "Gender",
        # PurposeOfLoan is categorical text. OtherLoansAtStore is removed to
        # match the repository's established German/GermanA preprocessing.
        "excluded_feature_columns": (
            "user_id",
            "GoodCustomer",
            "Gender",
            "PurposeOfLoan",
            "OtherLoansAtStore",
        ),
        "label_mapping": {"-1": 0, "1": 1},
        "sensitive_mapping": {"Male": 0, "Female": 1},
        "normalization": "none",
        "node_id_column": None,
        "target_feature_alignment": "exact",
        "unlabeled_label_value": None,
        "feature_file_has_header": True,
        "source_label_storage": "csv_column",
        "target_evaluation_storage": "csv_columns",
    },
    "pokec": {
        "dataset_family": "pokec",
        "source_domain": "pokec_z",
        "target_domain": "pokec_n",
        "label_column": "I_am_working_in_field",
        "sensitive_column": "region",
        # Pokec_z and Pokec_n have different one-hot vocabularies. Excluding
        # the union of domain-only fields leaves 264 shared non-sensitive
        # numeric features. Target columns are reordered to the source schema.
        "excluded_feature_columns": (
            "user_id",
            "I_am_working_in_field",
            "region",
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
        "label_mapping": {
            "-1": -1,
            "0": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 1,
        },
        "sensitive_mapping": {"0": 0, "1": 1},
        "normalization": "source_full_minmax_minus_one_one",
        "node_id_column": "user_id",
        "target_feature_alignment": "source_schema_zero_fill",
        "unlabeled_label_value": -1,
        "feature_file_has_header": True,
        "source_label_storage": "csv_column",
        "target_evaluation_storage": "csv_columns",
    },
    "syn": {
        "dataset_family": "syn",
        "source_domain": "syn-2",
        "target_domain": "syn-1",
        "label_column": "label",
        "sensitive_column": "sensitive",
        # Synthetic features are stored in a headerless 48-column numeric
        # matrix. Labels and sensitive values live in standalone text files
        # and are never appended to the model input.
        "excluded_feature_columns": (),
        "label_mapping": {"0": 0, "1": 1},
        "sensitive_mapping": {"0": 0, "1": 1},
        "normalization": "source_train_standardize",
        "node_id_column": None,
        "target_feature_alignment": "exact",
        "unlabeled_label_value": None,
        "feature_file_has_header": False,
        "source_label_storage": "separate_file",
        "target_evaluation_storage": "separate_files",
    },
}


def get_dataset_configuration(dataset_family):
    if dataset_family not in DATASET_CONFIGURATIONS:
        raise ValueError(
            "Unsupported dataset family {}. Choose one of {}".format(
                dataset_family, sorted(DATASET_CONFIGURATIONS)
            )
        )
    return DATASET_CONFIGURATIONS[dataset_family]


class GraphInputs(object):
    def __init__(self, features, adjacency, feature_names):
        self.features = features
        self.adjacency = adjacency
        self.feature_names = feature_names

    @property
    def num_nodes(self):
        return int(self.features.size(0))


def _read_header(csv_path):
    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        try:
            return next(reader)
        except StopIteration:
            raise ValueError("CSV file is empty: {}".format(csv_path))


def load_headerless_numeric_feature_matrix(
    csv_path,
    expected_feature_names=None,
    chunk_size=2048,
):
    """Load a headerless, entirely numeric feature matrix."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if expected_feature_names is None:
        feature_names = None
        expected_width = None
    else:
        feature_names = list(expected_feature_names)
        if not feature_names:
            raise ValueError("Expected feature names cannot be empty")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Expected feature names contain duplicates")
        expected_width = len(feature_names)

    chunks = []
    encoded_rows = []
    row_count = 0
    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if expected_width is None:
                expected_width = len(row)
                if expected_width == 0:
                    raise ValueError(
                        "No feature columns found in {}".format(csv_path)
                    )
                feature_names = [
                    "feature_{:03d}".format(index)
                    for index in range(expected_width)
                ]
            if len(row) != expected_width:
                raise ValueError(
                    "Feature width mismatch at {}:{}; expected {}, found {}".format(
                        csv_path, line_number, expected_width, len(row)
                    )
                )
            try:
                encoded_rows.append(
                    [float(value.strip()) for value in row]
                )
            except ValueError as error:
                raise ValueError(
                    "Invalid numeric feature at {}:{} ({})".format(
                        csv_path, line_number, error
                    )
                )
            row_count += 1
            if len(encoded_rows) >= chunk_size:
                chunks.append(
                    torch.tensor(encoded_rows, dtype=torch.float32)
                )
                encoded_rows = []

    if encoded_rows:
        chunks.append(torch.tensor(encoded_rows, dtype=torch.float32))
    if row_count == 0:
        raise ValueError("No data rows found in {}".format(csv_path))

    features = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    return features, feature_names


def load_feature_matrix(
    csv_path,
    expected_feature_names=None,
    excluded_feature_columns=EXCLUDED_FEATURE_COLUMNS,
    feature_alignment="exact",
    has_header=True,
):
    """Load model inputs while excluding configured protected columns."""

    if not has_header:
        if excluded_feature_columns not in (None, (), []):
            raise ValueError(
                "Headerless feature files cannot exclude named columns"
            )
        if feature_alignment != "exact":
            raise ValueError(
                "Headerless feature files require exact feature alignment"
            )
        return load_headerless_numeric_feature_matrix(
            csv_path,
            expected_feature_names=expected_feature_names,
        )

    header = _read_header(csv_path)
    excluded_feature_columns = tuple(excluded_feature_columns)
    discovered = [
        name for name in header if name not in excluded_feature_columns
    ]

    if expected_feature_names is None:
        feature_names = discovered
    else:
        feature_names = list(expected_feature_names)
        if feature_alignment == "exact" and discovered != feature_names:
            raise ValueError(
                "Feature schema mismatch in {}. Expected {}, found {}".format(
                    csv_path, feature_names, discovered
                )
            )
        if feature_alignment not in ("exact", "source_schema_zero_fill"):
            raise ValueError(
                "Unsupported feature alignment mode {}".format(
                    feature_alignment
                )
            )

    if not feature_names:
        raise ValueError("No usable feature columns found in {}".format(csv_path))

    column_to_index = {name: index for index, name in enumerate(header)}
    feature_indices = [column_to_index.get(name) for name in feature_names]
    rows = []

    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                rows.append(
                    [
                        0.0 if index is None else float(row[index])
                        for index in feature_indices
                    ]
                )
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "Invalid feature value at {}:{} ({})".format(
                        csv_path, line_number, error
                    )
                )

    if not rows:
        raise ValueError("No data rows found in {}".format(csv_path))

    return torch.tensor(rows, dtype=torch.float32), feature_names


def load_node_id_column(csv_path, column_name):
    """Load graph identifiers for edge re-indexing, never as model features."""

    header = _read_header(csv_path)
    if column_name not in header:
        raise ValueError("Missing column {} in {}".format(column_name, csv_path))
    column_index = header.index(column_name)
    node_ids = []
    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                node_id = int(float(row[column_index]))
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "Invalid {} at {}:{} ({})".format(
                        column_name, csv_path, line_number, error
                    )
                )
            node_ids.append(node_id)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Duplicate {} values in {}".format(column_name, csv_path))
    return node_ids


def _map_column_value(raw_value, value_mapping, column_name, csv_path, line_number):
    raw_value = raw_value.strip()
    if value_mapping is None:
        try:
            return int(float(raw_value))
        except ValueError as error:
            raise ValueError(
                "Invalid {} value at {}:{} ({})".format(
                    column_name, csv_path, line_number, error
                )
            )

    candidate_keys = [raw_value]
    try:
        numeric_value = float(raw_value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            candidate_keys.append(str(int(numeric_value)))
    except ValueError:
        pass

    for candidate in candidate_keys:
        if candidate in value_mapping:
            return int(value_mapping[candidate])
    raise ValueError(
        "Unmapped {} value {!r} at {}:{}; mapping keys are {}".format(
            column_name,
            raw_value,
            csv_path,
            line_number,
            sorted(value_mapping),
        )
    )


def load_label_column(csv_path, column_name="RECID", value_mapping=None):
    """Load labels explicitly and only for supervised source training."""

    header = _read_header(csv_path)
    if column_name not in header:
        raise ValueError("Missing column {} in {}".format(column_name, csv_path))
    column_index = header.index(column_name)
    values = []

    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                raw_value = row[column_index]
            except IndexError as error:
                raise ValueError(
                    "Invalid {} value at {}:{} ({})".format(
                        column_name, csv_path, line_number, error
                    )
                )
            values.append(
                _map_column_value(
                    raw_value,
                    value_mapping,
                    column_name,
                    csv_path,
                    line_number,
                )
            )

    return torch.tensor(values, dtype=torch.long)


def load_value_file(path, value_name="value", value_mapping=None):
    """Load one scalar value per line from a standalone text file."""

    values = []
    with open(path, "r") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) != 1:
                raise ValueError(
                    "Expected one {} at {}:{}, found {} values".format(
                        value_name, path, line_number, len(parts)
                    )
                )
            values.append(
                _map_column_value(
                    parts[0],
                    value_mapping,
                    value_name,
                    path,
                    line_number,
                )
            )
    if not values:
        raise ValueError("No {} values found in {}".format(value_name, path))
    return torch.tensor(values, dtype=torch.long)


def load_target_evaluation_columns(
    csv_path,
    label_column="RECID",
    sensitive_column="WHITE",
    label_mapping=None,
    sensitive_mapping=None,
):
    """Load target label/sensitive columns only in final evaluation."""

    header = _read_header(csv_path)
    for name in (label_column, sensitive_column):
        if name not in header:
            raise ValueError("Missing column {} in {}".format(name, csv_path))

    label_index = header.index(label_column)
    sensitive_index = header.index(sensitive_column)
    labels = []
    sensitive = []

    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                raw_label = row[label_index]
                raw_sensitive = row[sensitive_index]
            except IndexError as error:
                raise ValueError(
                    "Invalid evaluation value at {}:{} ({})".format(
                        csv_path, line_number, error
                    )
                )
            labels.append(
                _map_column_value(
                    raw_label,
                    label_mapping,
                    label_column,
                    csv_path,
                    line_number,
                )
            )
            sensitive.append(
                _map_column_value(
                    raw_sensitive,
                    sensitive_mapping,
                    sensitive_column,
                    csv_path,
                    line_number,
                )
            )

    return (
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(sensitive, dtype=torch.long),
    )


def load_target_evaluation_files(
    label_path,
    sensitive_path,
    label_name="label",
    sensitive_name="sensitive",
    label_mapping=None,
    sensitive_mapping=None,
):
    """Load standalone target labels/sensitivity only for final metrics."""

    labels = load_value_file(
        label_path,
        value_name=label_name,
        value_mapping=label_mapping,
    )
    sensitive = load_value_file(
        sensitive_path,
        value_name=sensitive_name,
        value_mapping=sensitive_mapping,
    )
    if labels.numel() != sensitive.numel():
        raise ValueError(
            "Target label and sensitive files have different lengths"
        )
    return labels, sensitive


def load_normalized_adjacency(edge_path, num_nodes, node_ids=None):
    """Return binary, symmetric, self-looped GCN-normalized adjacency."""

    sources = []
    targets = []
    with open(edge_path, "r") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(
                    "Invalid edge at {}:{}: {}".format(
                        edge_path, line_number, stripped
                    )
                )
            try:
                source_value = float(parts[0])
                target_value = float(parts[1])
            except ValueError as error:
                raise ValueError(
                    "Invalid edge at {}:{} ({})".format(
                        edge_path, line_number, error
                    )
                )
            if not math.isfinite(source_value) or not math.isfinite(target_value):
                raise ValueError(
                    "Non-finite edge at {}:{}: {}".format(
                        edge_path, line_number, stripped
                    )
                )
            source = int(round(source_value))
            target = int(round(target_value))
            if (
                abs(source_value - source) > 1e-8
                or abs(target_value - target) > 1e-8
            ):
                raise ValueError(
                    "Non-integral edge at {}:{}: {}".format(
                        edge_path, line_number, stripped
                    )
                )
            if (
                node_ids is None
                and (
                    source < 0
                    or source >= num_nodes
                    or target < 0
                    or target >= num_nodes
                )
            ):
                raise ValueError(
                    "Edge ({}, {}) at {}:{} is outside [0, {})".format(
                        source, target, edge_path, line_number, num_nodes
                    )
                )
            sources.append(source)
            targets.append(target)

    if not sources:
        raise ValueError("No edges found in {}".format(edge_path))

    if node_ids is not None:
        if len(node_ids) != num_nodes:
            raise ValueError("Node ID count does not match feature row count")
        id_to_row = {
            node_id: row_index for row_index, node_id in enumerate(node_ids)
        }
        edge_node_ids = set(sources)
        edge_node_ids.update(targets)
        missing_ids = sorted(edge_node_ids - set(id_to_row))
        if missing_ids:
            raise ValueError(
                "Edge file {} contains IDs absent from its CSV: {}{}".format(
                    edge_path,
                    missing_ids[:10],
                    " ..." if len(missing_ids) > 10 else "",
                )
            )
        sources = [id_to_row[node_id] for node_id in sources]
        targets = [id_to_row[node_id] for node_id in targets]

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    reverse_index = torch.stack((edge_index[1], edge_index[0]), dim=0)
    node_index = torch.arange(num_nodes, dtype=torch.long)
    loop_index = torch.stack((node_index, node_index), dim=0)
    all_indices = torch.cat((edge_index, reverse_index, loop_index), dim=1)

    unweighted = torch.sparse_coo_tensor(
        all_indices,
        torch.ones(all_indices.size(1), dtype=torch.float32),
        torch.Size((num_nodes, num_nodes)),
    ).coalesce()

    # Coalescing sums duplicate/reverse/self edges. Rebuild with unit weights.
    unique_indices = unweighted.indices()
    binary_adjacency = torch.sparse_coo_tensor(
        unique_indices,
        torch.ones(unique_indices.size(1), dtype=torch.float32),
        torch.Size((num_nodes, num_nodes)),
    ).coalesce()

    degree = torch.sparse.sum(binary_adjacency, dim=1).to_dense()
    inverse_sqrt_degree = degree.clamp(min=1.0).pow(-0.5)
    row, column = binary_adjacency.indices()
    normalized_values = (
        inverse_sqrt_degree[row]
        * binary_adjacency.values()
        * inverse_sqrt_degree[column]
    )

    return torch.sparse_coo_tensor(
        binary_adjacency.indices(),
        normalized_values,
        torch.Size((num_nodes, num_nodes)),
    ).coalesce()


def load_graph_inputs(
    csv_path,
    edge_path,
    expected_feature_names=None,
    excluded_feature_columns=EXCLUDED_FEATURE_COLUMNS,
    feature_alignment="exact",
    node_id_column=None,
    feature_file_has_header=True,
):
    features, feature_names = load_feature_matrix(
        csv_path,
        expected_feature_names=expected_feature_names,
        excluded_feature_columns=excluded_feature_columns,
        feature_alignment=feature_alignment,
        has_header=feature_file_has_header,
    )
    if not feature_file_has_header and node_id_column is not None:
        raise ValueError(
            "Headerless feature files cannot provide a named node ID column"
        )
    node_ids = (
        load_node_id_column(csv_path, node_id_column)
        if node_id_column is not None
        else None
    )
    adjacency = load_normalized_adjacency(
        edge_path, features.size(0), node_ids=node_ids
    )
    return GraphInputs(features, adjacency, feature_names)


def compute_feature_statistics(features, train_mask):
    selected = features[train_mask]
    if selected.size(0) == 0:
        raise ValueError("Cannot compute feature statistics from an empty mask")
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp(min=1e-6)
    return mean, std


def standardize_features(features, mean, std):
    if features.size(1) != mean.numel() or mean.numel() != std.numel():
        raise ValueError("Feature normalization dimensions do not match")
    return (features - mean.view(1, -1)) / std.view(1, -1)
