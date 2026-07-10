from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def encode_ys_groups(y, sens) -> np.ndarray:
    """Encode binary (Y, S) pairs with the same convention used by SFFGNN."""
    y_arr = np.asarray(y).astype(int).reshape(-1)
    sens_arr = np.asarray(sens).astype(int).reshape(-1)
    if y_arr.shape[0] != sens_arr.shape[0]:
        raise ValueError("y and sens must have the same length.")

    labels = np.full(y_arr.shape[0], -1, dtype=np.int64)
    labels[(y_arr == 1) & (sens_arr == 0)] = 0
    labels[(y_arr == 1) & (sens_arr == 1)] = 1
    labels[(y_arr == 0) & (sens_arr == 0)] = 2
    labels[(y_arr == 0) & (sens_arr == 1)] = 3
    return labels


def save_visualization_embeddings(
    output_dir,
    dataset: str,
    representations,
    *,
    labels: Optional[object] = None,
    y: Optional[object] = None,
    sens: Optional[object] = None,
    stage: Optional[str] = "source_trained",
) -> tuple[Path, Path]:
    """Save embeddings in the standard format consumed by run_tsne.py.

    Baselines can import this helper, or simply write equivalent npz files:
    - {dataset}_{stage}_feat.npz with key "representations"
    - {dataset}_{stage}_labels.npz with key "labels"
    """
    emb = np.asarray(representations)
    if emb.ndim != 2:
        raise ValueError("representations must be a 2D array shaped [num_nodes, dim].")

    if labels is None:
        if y is None or sens is None:
            raise ValueError("Provide either labels or both y and sens.")
        labels_arr = encode_ys_groups(y, sens)
    else:
        labels_arr = np.asarray(labels).astype(int).reshape(-1)

    if emb.shape[0] != labels_arr.shape[0]:
        raise ValueError(
            f"representations and labels length mismatch: {emb.shape[0]} vs {labels_arr.shape[0]}."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{stage}" if stage else ""
    feat_path = out_dir / f"{dataset}{suffix}_feat.npz"
    labels_path = out_dir / f"{dataset}{suffix}_labels.npz"
    np.savez_compressed(feat_path, representations=emb)
    np.savez_compressed(labels_path, labels=labels_arr)
    return feat_path, labels_path

