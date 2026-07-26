"""Data utilities for binary graph source-free adaptation experiments.

The feature loader accepts configurable identifier, label, and sensitive
columns. Source labels and target evaluation columns are loaded through
separate functions so target adaptation cannot accidentally consume them.
"""

from __future__ import print_function

import csv

import torch


EXCLUDED_FEATURE_COLUMNS = ("user_id", "RECID", "WHITE")


class GraphInputs(object):
    def __init__(self, features, adjacency, feature_names, feature_schema):
        self.features = features
        self.adjacency = adjacency
        self.feature_names = feature_names
        self.feature_schema = feature_schema

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


def _can_convert_to_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _parse_integer(value):
    """Parse integer-looking CSV/edge values, including scientific notation."""

    return int(float(value))


def _column_index_map(header, csv_path):
    column_to_index = {}
    duplicates = []
    for index, name in enumerate(header):
        if name in column_to_index:
            duplicates.append(name)
        else:
            column_to_index[name] = index
    if duplicates:
        raise ValueError(
            "Duplicate CSV columns in {}: {}".format(
                csv_path, sorted(set(duplicates))
            )
        )
    return column_to_index


def load_headerless_numeric_feature_matrix(
    csv_path,
    expected_feature_names=None,
    expected_feature_schema=None,
    chunk_size=2048,
):
    """Load a headerless, entirely numeric feature matrix in bounded chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if expected_feature_schema is None:
        feature_schema = None
        feature_names = None
        expected_width = None
    else:
        feature_schema = [dict(entry) for entry in expected_feature_schema]
        if not feature_schema:
            raise ValueError("Expected feature schema cannot be empty")
        if any(entry.get("kind") != "numeric" for entry in feature_schema):
            raise ValueError("Headerless feature files support numeric columns only")
        feature_names = [entry["name"] for entry in feature_schema]
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Expected feature schema contains duplicate names")
        expected_width = len(feature_schema)

    chunks = []
    encoded_rows = []
    row_count = 0
    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if expected_width is None:
                expected_width = len(row)
                if expected_width == 0:
                    raise ValueError("No feature columns found in {}".format(csv_path))
                feature_names = [
                    "feature_{:03d}".format(index)
                    for index in range(expected_width)
                ]
                feature_schema = [
                    {"name": name, "kind": "numeric"}
                    for name in feature_names
                ]
            if len(row) != expected_width:
                raise ValueError(
                    "Feature width mismatch at {}:{}; expected {}, found {}".format(
                        csv_path, row_number, expected_width, len(row)
                    )
                )
            try:
                encoded_rows.append([float(value.strip()) for value in row])
            except ValueError as error:
                raise ValueError(
                    "Invalid numeric feature at {}:{} ({})".format(
                        csv_path, row_number, error
                    )
                )
            row_count += 1
            if len(encoded_rows) >= chunk_size:
                chunks.append(torch.tensor(encoded_rows, dtype=torch.float32))
                encoded_rows = []

    if encoded_rows:
        chunks.append(torch.tensor(encoded_rows, dtype=torch.float32))
    if row_count == 0:
        raise ValueError("No data rows found in {}".format(csv_path))

    if expected_feature_names is not None:
        expected_feature_names = list(expected_feature_names)
        if feature_names != expected_feature_names:
            raise ValueError(
                "Headerless feature mismatch in {}. Expected {}, found {}".format(
                    csv_path, expected_feature_names, feature_names
                )
            )

    features = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    return features, feature_names, feature_schema


def load_feature_matrix(
    csv_path,
    expected_feature_names=None,
    excluded_feature_columns=None,
    expected_feature_schema=None,
    align_to_expected_schema=False,
    chunk_size=2048,
    has_header=True,
):
    """Load model inputs while ignoring configured special columns.

    The label and sensitive columns are not converted to tensors or returned
    from this function. Calling it during target adaptation therefore does not
    make target labels or sensitive groups available to the optimizer.
    """

    if not has_header:
        if excluded_feature_columns not in (None, (), []):
            raise ValueError(
                "Headerless feature files cannot contain excluded named columns"
            )
        return load_headerless_numeric_feature_matrix(
            csv_path,
            expected_feature_names=expected_feature_names,
            expected_feature_schema=expected_feature_schema,
            chunk_size=chunk_size,
        )

    if excluded_feature_columns is None:
        excluded_feature_columns = EXCLUDED_FEATURE_COLUMNS
    excluded_feature_columns = tuple(excluded_feature_columns)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    header = _read_header(csv_path)
    column_to_index = _column_index_map(header, csv_path)
    missing_excluded = [
        name for name in excluded_feature_columns if name not in column_to_index
    ]
    if missing_excluded:
        raise ValueError(
            "Configured excluded columns are missing from {}: {}".format(
                csv_path, missing_excluded
            )
        )
    discovered = [
        name for name in header if name not in excluded_feature_columns
    ]
    if not discovered:
        raise ValueError("No usable feature columns found in {}".format(csv_path))

    if expected_feature_schema is None:
        selected_names = list(discovered)
        feature_indices = [column_to_index[name] for name in selected_names]
        numeric_columns = [True] * len(selected_names)
        observed_values = [set() for _ in selected_names]
        row_count = 0
        with open(csv_path, "r", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        "CSV width mismatch at {}:{}; expected {}, found {}".format(
                            csv_path, line_number, len(header), len(row)
                        )
                    )
                row_count += 1
                for offset, column_index in enumerate(feature_indices):
                    value = row[column_index].strip()
                    observed_values[offset].add(value)
                    if numeric_columns[offset] and not _can_convert_to_float(value):
                        numeric_columns[offset] = False
        if row_count == 0:
            raise ValueError("No data rows found in {}".format(csv_path))

        feature_schema = []
        for offset, name in enumerate(selected_names):
            if numeric_columns[offset]:
                feature_schema.append({"name": name, "kind": "numeric"})
            else:
                categories = sorted(observed_values[offset])
                feature_schema.append(
                    {
                        "name": name,
                        "kind": "categorical",
                        "categories": categories,
                    }
                )
    else:
        feature_schema = [dict(entry) for entry in expected_feature_schema]
        expected_base_names = [entry["name"] for entry in feature_schema]
        if len(set(expected_base_names)) != len(expected_base_names):
            raise ValueError("Expected feature schema contains duplicate names")
        if not align_to_expected_schema and discovered != expected_base_names:
            raise ValueError(
                "Feature schema mismatch in {}. Expected {}, found {}".format(
                    csv_path, expected_base_names, discovered
                )
            )
        selected_names = expected_base_names
        feature_indices = [column_to_index.get(name) for name in selected_names]

    feature_names = []
    for entry in feature_schema:
        if entry["kind"] == "numeric":
            feature_names.append(entry["name"])
        elif entry["kind"] == "categorical":
            for category in entry["categories"]:
                feature_names.append("{}={}".format(entry["name"], category))
        else:
            raise ValueError(
                "Unknown feature kind {} for {}".format(
                    entry["kind"], entry["name"]
                )
            )

    if expected_feature_names is not None:
        expected_feature_names = list(expected_feature_names)
        if feature_names != expected_feature_names:
            raise ValueError(
                "Expanded feature mismatch in {}. Expected {}, found {}".format(
                    csv_path, expected_feature_names, feature_names
                )
            )

    chunks = []
    encoded_rows = []
    row_count = 0
    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    "CSV width mismatch at {}:{}; expected {}, found {}".format(
                        csv_path, row_number, len(header), len(row)
                    )
                )
            encoded = []
            for column_index, entry in zip(feature_indices, feature_schema):
                if column_index is None:
                    if not align_to_expected_schema:
                        raise ValueError(
                            "Missing expected feature {} in {}".format(
                                entry["name"], csv_path
                            )
                        )
                    if entry["kind"] == "numeric":
                        encoded.append(0.0)
                    elif entry["kind"] == "categorical":
                        encoded.extend([0.0] * len(entry["categories"]))
                    else:
                        raise ValueError(
                            "Unknown feature kind {} for {}".format(
                                entry["kind"], entry["name"]
                            )
                        )
                    continue

                value = row[column_index].strip()
                if entry["kind"] == "numeric":
                    try:
                        encoded.append(float(value))
                    except ValueError as error:
                        raise ValueError(
                            "Invalid numeric feature {} at {}:{} ({})".format(
                                entry["name"], csv_path, row_number, error
                            )
                        )
                elif entry["kind"] == "categorical":
                    categories = entry["categories"]
                    if value not in categories:
                        raise ValueError(
                            "Unknown category {} for {} at {}:{}".format(
                                value, entry["name"], csv_path, row_number
                            )
                        )
                    encoded.extend(
                        [1.0 if value == category else 0.0 for category in categories]
                    )
                else:
                    raise ValueError(
                        "Unknown feature kind {} for {}".format(
                            entry["kind"], entry["name"]
                        )
                    )

            encoded_rows.append(encoded)
            row_count += 1
            if len(encoded_rows) >= chunk_size:
                chunks.append(torch.tensor(encoded_rows, dtype=torch.float32))
                encoded_rows = []

    if encoded_rows:
        chunks.append(torch.tensor(encoded_rows, dtype=torch.float32))
    if row_count == 0:
        raise ValueError("No data rows found in {}".format(csv_path))
    features = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    return features, feature_names, feature_schema


def load_identifier_column(csv_path, column_name="user_id"):
    """Load original node identifiers for edge-list reindexing."""

    header = _read_header(csv_path)
    column_to_index = _column_index_map(header, csv_path)
    if column_name not in column_to_index:
        raise ValueError("Missing column {} in {}".format(column_name, csv_path))
    column_index = column_to_index[column_name]
    identifiers = []
    seen = set()
    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                identifier = _parse_integer(row[column_index].strip())
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "Invalid {} at {}:{} ({})".format(
                        column_name, csv_path, line_number, error
                    )
                )
            if identifier in seen:
                raise ValueError(
                    "Duplicate {}={} at {}:{}".format(
                        column_name, identifier, csv_path, line_number
                    )
                )
            seen.add(identifier)
            identifiers.append(identifier)
    if not identifiers:
        raise ValueError("No identifiers found in {}".format(csv_path))
    return identifiers


def load_label_column(
    csv_path,
    column_name="RECID",
    value_mapping=None,
    ignored_values=None,
    ignore_index=-1,
):
    """Load a label column explicitly for supervised source training."""

    header = _read_header(csv_path)
    if column_name not in header:
        raise ValueError("Missing column {} in {}".format(column_name, csv_path))
    column_index = header.index(column_name)
    values = []
    ignored_values = set(str(value) for value in (ignored_values or []))

    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                raw_value = row[column_index].strip()
                if raw_value in ignored_values:
                    values.append(int(ignore_index))
                elif value_mapping is None:
                    values.append(_parse_integer(raw_value))
                else:
                    if raw_value not in value_mapping:
                        raise ValueError(
                            "Unknown {} value {}".format(column_name, raw_value)
                        )
                    values.append(int(value_mapping[raw_value]))
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "Invalid {} value at {}:{} ({})".format(
                        column_name, csv_path, line_number, error
                    )
                )

    return torch.tensor(values, dtype=torch.long)


def load_value_file(
    path,
    value_name="value",
    value_mapping=None,
    ignored_values=None,
    ignore_index=-1,
):
    """Load one scalar value per line from a standalone text file."""

    values = []
    ignored_values = set(str(value) for value in (ignored_values or []))
    with open(path, "r") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) != 1:
                raise ValueError(
                    "Expected one {} value at {}:{}, found {}".format(
                        value_name, path, line_number, len(parts)
                    )
                )
            raw_value = parts[0]
            try:
                if raw_value in ignored_values:
                    values.append(int(ignore_index))
                elif value_mapping is None:
                    values.append(_parse_integer(raw_value))
                else:
                    if raw_value not in value_mapping:
                        raise ValueError(
                            "Unknown {} value {}".format(value_name, raw_value)
                        )
                    values.append(int(value_mapping[raw_value]))
            except ValueError as error:
                raise ValueError(
                    "Invalid {} at {}:{} ({})".format(
                        value_name, path, line_number, error
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
    ignored_label_values=None,
    ignore_index=-1,
):
    """Load target labels and sensitivity only after adaptation completes."""

    header = _read_header(csv_path)
    for name in (label_column, sensitive_column):
        if name not in header:
            raise ValueError("Missing column {} in {}".format(name, csv_path))

    label_index = header.index(label_column)
    sensitive_index = header.index(sensitive_column)
    labels = []
    sensitive = []
    ignored_label_values = set(
        str(value) for value in (ignored_label_values or [])
    )

    with open(csv_path, "r", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                raw_label = row[label_index].strip()
                raw_sensitive = row[sensitive_index].strip()
                if raw_label in ignored_label_values:
                    labels.append(int(ignore_index))
                elif label_mapping is None:
                    labels.append(_parse_integer(raw_label))
                else:
                    if raw_label not in label_mapping:
                        raise ValueError(
                            "Unknown {} value {}".format(label_column, raw_label)
                        )
                    labels.append(int(label_mapping[raw_label]))
                if sensitive_mapping is None:
                    sensitive.append(_parse_integer(raw_sensitive))
                else:
                    if raw_sensitive not in sensitive_mapping:
                        raise ValueError(
                            "Unknown {} value {}".format(
                                sensitive_column, raw_sensitive
                            )
                        )
                    sensitive.append(int(sensitive_mapping[raw_sensitive]))
            except (ValueError, IndexError) as error:
                raise ValueError(
                    "Invalid evaluation value at {}:{} ({})".format(
                        csv_path, line_number, error
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
    ignored_label_values=None,
    ignore_index=-1,
):
    """Load standalone target evaluation files after adaptation completes."""

    labels = load_value_file(
        label_path,
        value_name=label_name,
        value_mapping=label_mapping,
        ignored_values=ignored_label_values,
        ignore_index=ignore_index,
    )
    sensitive = load_value_file(
        sensitive_path,
        value_name=sensitive_name,
        value_mapping=sensitive_mapping,
    )
    if labels.numel() != sensitive.numel():
        raise ValueError("Target label and sensitive files have different lengths")
    return labels, sensitive


def load_normalized_adjacency(edge_path, num_nodes, node_identifiers=None):
    """Load an edge list and return symmetric GCN-normalized sparse adjacency.

    The loader symmetrizes and adds loops defensively, then binarizes after
    coalescing so pre-existing reverse edges/loops are not double weighted.
    """

    identifier_to_row = None
    if node_identifiers is not None:
        if len(node_identifiers) != num_nodes:
            raise ValueError("Identifier count does not match the node count")
        identifier_to_row = {}
        for row_index, identifier in enumerate(node_identifiers):
            if identifier in identifier_to_row:
                raise ValueError("Duplicate node identifier {}".format(identifier))
            identifier_to_row[identifier] = row_index

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
                raw_source = _parse_integer(parts[0])
                raw_target = _parse_integer(parts[1])
            except ValueError as error:
                raise ValueError(
                    "Invalid edge at {}:{} ({})".format(
                        edge_path, line_number, error
                    )
                )
            if identifier_to_row is None:
                source = raw_source
                target = raw_target
                if (
                    source < 0
                    or source >= num_nodes
                    or target < 0
                    or target >= num_nodes
                ):
                    raise ValueError(
                        "Edge ({}, {}) at {}:{} is outside [0, {})".format(
                            source, target, edge_path, line_number, num_nodes
                        )
                    )
            else:
                try:
                    source = identifier_to_row[raw_source]
                    target = identifier_to_row[raw_target]
                except KeyError as error:
                    raise ValueError(
                        "Edge endpoint {} at {}:{} is absent from the CSV "
                        "identifier column".format(
                            error.args[0], edge_path, line_number
                        )
                    )
            sources.append(source)
            targets.append(target)

    if not sources:
        raise ValueError("No edges found in {}".format(edge_path))

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

    # Coalescing sums duplicates. Rebuild with ones to make the graph binary.
    unique_indices = unweighted.indices()
    binary_values = torch.ones(unique_indices.size(1), dtype=torch.float32)
    binary_adjacency = torch.sparse_coo_tensor(
        unique_indices,
        binary_values,
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
    excluded_feature_columns=None,
    expected_feature_schema=None,
    id_column=None,
    align_to_expected_schema=False,
    has_header=True,
):
    features, feature_names, feature_schema = load_feature_matrix(
        csv_path,
        expected_feature_names=expected_feature_names,
        excluded_feature_columns=excluded_feature_columns,
        expected_feature_schema=expected_feature_schema,
        align_to_expected_schema=align_to_expected_schema,
        has_header=has_header,
    )
    node_identifiers = None
    if id_column is not None:
        node_identifiers = load_identifier_column(csv_path, id_column)
        if len(node_identifiers) != features.size(0):
            raise ValueError("Feature and identifier row counts do not match")
    adjacency = load_normalized_adjacency(
        edge_path,
        features.size(0),
        node_identifiers=node_identifiers,
    )
    return GraphInputs(features, adjacency, feature_names, feature_schema)


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
