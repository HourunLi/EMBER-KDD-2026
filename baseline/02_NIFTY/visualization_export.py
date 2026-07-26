"""Validated target-embedding export shared by the local GCN baselines.

The output layout follows ``visualization/`` on the ``zyt`` branch of
HourunLi/AAAI-2026.  A seed is represented as part of the method name so that
every run still has the standard ``<method>/<dataset>/feat.npz`` layout while
never overwriting another seed.
"""

from pathlib import Path
from typing import Tuple

import numpy as np


def method_name_for_seed(method: str, seed: int) -> str:
    """Return a visualization method name that uniquely identifies one seed."""

    method = method.strip()
    if not method:
        raise ValueError("Visualization method name must not be empty.")
    return f"{method}_seed{int(seed)}"


def encode_predicted_ys_groups(
    predictions: np.ndarray,
    sensitive: np.ndarray,
) -> np.ndarray:
    """Encode predicted-Y/sensitive groups using the zyt convention.

    The required mapping is::

        0 -> predicted Y=1, S=0
        1 -> predicted Y=1, S=1
        2 -> predicted Y=0, S=0
        3 -> predicted Y=0, S=1

    Equivalently, ``group = 2 * (1 - predicted_y) + sensitive``.
    """

    predicted_y = np.asarray(predictions).reshape(-1)
    sens = np.asarray(sensitive).reshape(-1)
    if predicted_y.shape[0] != sens.shape[0]:
        raise ValueError(
            "Predictions and sensitive attributes must have the same length."
        )

    predicted_values = set(np.unique(predicted_y).tolist())
    if not predicted_values.issubset({0, 1}):
        raise ValueError(
            "Target predictions used for visualization must be binary 0/1, "
            f"got {sorted(predicted_values)}."
        )
    sensitive_values = set(np.unique(sens).tolist())
    if not sensitive_values.issubset({0, 1}):
        raise ValueError(
            "Target sensitive attributes used for visualization must be "
            f"binary 0/1, got {sorted(sensitive_values)}."
        )

    predicted_y = predicted_y.astype(np.int64, copy=False)
    sens = sens.astype(np.int64, copy=False)
    labels = 2 * (1 - predicted_y) + sens
    label_values = set(np.unique(labels).tolist())
    if not label_values.issubset({0, 1, 2, 3}):
        raise AssertionError(
            f"Encoded visualization labels are outside 0/1/2/3: {label_values}."
        )
    return labels.astype(np.int64, copy=False)


def save_target_visualization_embeddings(
    embeddings_root: Path,
    method: str,
    dataset: str,
    representations: np.ndarray,
    predictions: np.ndarray,
    sensitive: np.ndarray,
) -> Tuple[Path, Path]:
    """Validate and save one seed's frozen-target representations and groups."""

    embeddings = np.asarray(representations)
    if embeddings.ndim != 2:
        raise ValueError(
            "representations must be a 2D array shaped "
            "[num_valid_target_all_nodes, feature_dim]."
        )
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot export an empty target representation array.")
    try:
        all_finite = bool(np.isfinite(embeddings).all())
    except TypeError as error:
        raise ValueError("representations must contain numeric values.") from error
    if not all_finite:
        raise ValueError("representations contains NaN or Inf.")

    labels = encode_predicted_ys_groups(predictions, sensitive)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "representations and labels length mismatch: "
            f"{embeddings.shape[0]} vs {labels.shape[0]}."
        )

    dataset = dataset.strip()
    if not dataset:
        raise ValueError("Visualization dataset name must not be empty.")
    output_dir = Path(embeddings_root) / method / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_path = output_dir / "feat.npz"
    labels_path = output_dir / "labels.npz"
    np.savez_compressed(feat_path, representations=embeddings)
    np.savez_compressed(labels_path, labels=labels)
    return feat_path, labels_path
