from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .export_utils import encode_ys_groups
except ImportError:
    from export_utils import encode_ys_groups


DEFAULT_GROUP_NAMES = {
    0: "Y=1, S=0",
    1: "Y=1, S=1",
    2: "Y=0, S=0",
    3: "Y=0, S=1",
}


@dataclass
class LoadedEmbedding:
    method: str
    dataset: str
    embeddings: np.ndarray
    labels: np.ndarray
    group_names: dict[int, str]
    embedding_path: Path
    labels_path: Path | None


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read visualization/config.yaml. "
            "Install it with: pip install pyyaml"
        ) from exc

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return data


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def split_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unnamed"


def get_group_names(config: Mapping[str, Any]) -> dict[int, str]:
    raw_groups = ((config.get("labels") or {}).get("groups")) or {}
    groups = DEFAULT_GROUP_NAMES.copy()
    for key, value in raw_groups.items():
        groups[int(key)] = str(value)
    return groups


def enabled_names(entries: Sequence[Any], explicit: Sequence[str] | None = None) -> list[str]:
    if explicit:
        return list(explicit)

    names: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, Mapping):
            if entry.get("enabled", True):
                names.append(str(entry["name"]))
    return names


def dataset_lookup(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in config.get("datasets", []):
        if isinstance(item, str):
            lookup[item] = {"name": item, "title": item}
        elif isinstance(item, Mapping):
            name = str(item["name"])
            lookup[name] = dict(item)
    return lookup


def method_lookup(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in config.get("methods", []):
        if isinstance(item, str):
            lookup[item] = {"name": item}
        elif isinstance(item, Mapping):
            name = str(item["name"])
            lookup[name] = dict(item)
    return lookup


def build_context(
    config: Mapping[str, Any],
    config_path: Path,
    dataset: str,
    method_name: str,
) -> dict[str, str]:
    config_dir = config_path.resolve().parent
    project_root = resolve_path(
        ((config.get("project") or {}).get("root")) or str(project_root_from_here()),
        {"config_dir": str(config_dir)},
        config_dir,
    )
    visualization_root = project_root / "visualization"

    return {
        "config_dir": str(config_dir),
        "project_root": str(project_root),
        "visualization_root": str(visualization_root),
        "embeddings_root": str(visualization_root / "embeddings"),
        "results_root": str(visualization_root / "results"),
        "dataset": dataset,
        "method": method_name,
        "method_name": method_name,
    }


def resolve_path(value: str, context: Mapping[str, str], base_dir: Path) -> Path:
    formatted = str(value).format(**context)
    formatted = os.path.expanduser(os.path.expandvars(formatted))
    path = Path(formatted)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _path_templates(
    method_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    key: str,
    default_key: str,
) -> list[str]:
    for candidate_key in (key, f"{key}s"):
        if candidate_key in method_cfg:
            value = method_cfg[candidate_key]
            return list(value) if isinstance(value, list) else [str(value)]

    defaults = config.get("defaults") or {}
    value = defaults.get(default_key) or defaults.get(f"{default_key}s") or []
    if isinstance(value, list):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return []


def _first_existing_path(
    templates: Sequence[str],
    context: Mapping[str, str],
    base_dir: Path,
) -> tuple[Path | None, list[Path]]:
    checked: list[Path] = []
    for template in templates:
        path = resolve_path(template, context, base_dir)
        checked.append(path)
        if path.exists():
            return path, checked
    return None, checked


def load_method_dataset(
    config: Mapping[str, Any],
    config_path: Path,
    method_cfg: Mapping[str, Any],
    dataset: str,
) -> LoadedEmbedding:
    config_dir = config_path.resolve().parent
    method_name = str(method_cfg["name"])
    context = build_context(config, config_path, dataset, method_name)

    embedding_templates = _path_templates(method_cfg, config, "embedding_path", "embedding_paths")
    label_templates = _path_templates(method_cfg, config, "label_path", "label_paths")
    if not embedding_templates:
        raise ValueError(f"No embedding_path configured for method {method_name}.")

    embedding_path, checked_embeddings = _first_existing_path(
        embedding_templates, context, config_dir
    )
    if embedding_path is None:
        checked = "\n  - ".join(str(path) for path in checked_embeddings)
        raise FileNotFoundError(
            f"Missing embeddings for method={method_name}, dataset={dataset}.\n"
            f"Checked:\n  - {checked}"
        )

    labels_path, _ = _first_existing_path(label_templates, context, config_dir)
    embedding_keys = _as_key_list(
        method_cfg.get("embedding_key"), ["representations", "embeddings", "features", "x"]
    )
    emb = load_array(embedding_path, embedding_keys)
    labels = load_labels(labels_path, embedding_path, method_cfg)

    emb, labels = clean_and_validate(emb, labels, method_name, dataset)
    return LoadedEmbedding(
        method=method_name,
        dataset=dataset,
        embeddings=emb,
        labels=labels,
        group_names=get_group_names(config),
        embedding_path=embedding_path,
        labels_path=labels_path,
    )


def _as_key_list(value: Any, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_array(path: Path, keys: Sequence[str]) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in keys:
                if key in data.files:
                    return np.asarray(data[key])
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])
            raise KeyError(
                f"{path} does not contain any of keys {list(keys)}. Available keys: {data.files}"
            )
    raise ValueError(f"Unsupported array file type: {path}")


def load_labels(labels_path: Path | None, embedding_path: Path, method_cfg: Mapping[str, Any]) -> np.ndarray:
    label_keys = _as_key_list(method_cfg.get("label_key"), ["labels", "label", "groups", "ys_group"])
    y_keys = _as_key_list(method_cfg.get("y_key"), ["y", "labels_y", "target", "targets"])
    sens_keys = _as_key_list(method_cfg.get("sens_key"), ["sens", "s", "sensitive", "sens_labels"])

    if labels_path is not None:
        return load_label_array_or_encode(labels_path, label_keys, y_keys, sens_keys)

    if embedding_path.suffix.lower() == ".npz":
        return load_label_array_or_encode(embedding_path, label_keys, y_keys, sens_keys)

    raise FileNotFoundError(
        f"No labels_path found for {embedding_path}. Provide labels.npz or include labels/y+sens in feat.npz."
    )


def load_label_array_or_encode(
    path: Path,
    label_keys: Sequence[str],
    y_keys: Sequence[str],
    sens_keys: Sequence[str],
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported label file type: {path}")

    with np.load(path, allow_pickle=False) as data:
        for key in label_keys:
            if key in data.files:
                return np.asarray(data[key]).reshape(-1)

        y_key = next((key for key in y_keys if key in data.files), None)
        sens_key = next((key for key in sens_keys if key in data.files), None)
        if y_key and sens_key:
            return encode_ys_groups(data[y_key], data[sens_key])

        raise KeyError(
            f"{path} does not contain labels or y/sens arrays. Available keys: {data.files}"
        )


def clean_and_validate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    emb = np.asarray(embeddings, dtype=np.float64)
    lab = np.asarray(labels).astype(int).reshape(-1)

    if emb.ndim != 2:
        raise ValueError(f"{method}/{dataset}: embeddings must be 2D, got shape {emb.shape}.")
    if emb.shape[0] != lab.shape[0]:
        raise ValueError(
            f"{method}/{dataset}: embeddings and labels length mismatch: {emb.shape[0]} vs {lab.shape[0]}."
        )

    finite_mask = np.isfinite(emb).all(axis=1) & np.isfinite(lab) & (lab >= 0)
    if not finite_mask.any():
        raise ValueError(f"{method}/{dataset}: no valid rows after filtering.")
    return emb[finite_mask], lab[finite_mask]


def sample_points(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_points: int | None,
    *,
    stratify: bool,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = labels.shape[0]
    if not max_points or max_points <= 0 or n <= max_points:
        idx = np.arange(n)
        return embeddings, labels, idx

    rng = np.random.default_rng(random_state)
    if not stratify:
        idx = rng.choice(n, size=max_points, replace=False)
        idx.sort()
        return embeddings[idx], labels[idx], idx

    selected: list[int] = []
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_idx = np.flatnonzero(labels == label)
        quota = max(1, int(round(max_points * count / n)))
        quota = min(quota, label_idx.shape[0])
        selected.extend(rng.choice(label_idx, size=quota, replace=False).tolist())

    if len(selected) > max_points:
        selected = rng.choice(np.asarray(selected), size=max_points, replace=False).tolist()
    elif len(selected) < max_points:
        remaining = np.setdiff1d(np.arange(n), np.asarray(selected), assume_unique=False)
        fill = min(max_points - len(selected), remaining.shape[0])
        if fill > 0:
            selected.extend(rng.choice(remaining, size=fill, replace=False).tolist())

    idx = np.asarray(selected, dtype=int)
    idx.sort()
    return embeddings[idx], labels[idx], idx


def write_coordinates_csv(
    path: Path,
    method: str,
    dataset: str,
    coords: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    group_names: Mapping[int, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dataset", "source_index", "x", "y", "group", "group_name"])
        for idx, xy, label in zip(source_indices, coords, labels):
            label_int = int(label)
            writer.writerow(
                [
                    method,
                    dataset,
                    int(idx),
                    float(xy[0]),
                    float(xy[1]),
                    label_int,
                    group_names.get(label_int, str(label_int)),
                ]
            )
