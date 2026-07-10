from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc
    return plt


def compute_tsne(embeddings: np.ndarray, tsne_cfg: Mapping[str, Any]) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for t-SNE. Install it with: pip install scikit-learn"
        ) from exc

    n = embeddings.shape[0]
    if n < 3:
        raise ValueError("t-SNE requires at least 3 points.")

    requested_perplexity = float(tsne_cfg.get("perplexity", 30))
    perplexity = min(requested_perplexity, max(1.0, (n - 1) / 3.0))

    params = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": tsne_cfg.get("init", "pca"),
        "learning_rate": tsne_cfg.get("learning_rate", "auto"),
        "metric": tsne_cfg.get("metric", "euclidean"),
        "random_state": int(tsne_cfg.get("random_state", 42)),
    }

    signature = inspect.signature(TSNE)
    max_iter = int(tsne_cfg.get("max_iter", tsne_cfg.get("n_iter", 1000)))
    if "max_iter" in signature.parameters:
        params["max_iter"] = max_iter
    else:
        params["n_iter"] = max_iter

    if "n_jobs" in signature.parameters and "n_jobs" in tsne_cfg:
        params["n_jobs"] = int(tsne_cfg["n_jobs"])

    coords = TSNE(**params).fit_transform(embeddings)
    if tsne_cfg.get("normalize", "minmax") == "minmax":
        coords = minmax_normalize(coords)
    return coords


def minmax_normalize(coords: np.ndarray) -> np.ndarray:
    min_vals = coords.min(axis=0, keepdims=True)
    max_vals = coords.max(axis=0, keepdims=True)
    denom = max_vals - min_vals
    scaled = np.zeros_like(coords, dtype=np.float64)
    non_constant = denom.reshape(-1) > 1e-12
    scaled[:, non_constant] = (coords[:, non_constant] - min_vals[:, non_constant]) / denom[:, non_constant]
    scaled[:, ~non_constant] = 0.5
    return scaled


def plot_panels(
    output_path: Path,
    dataset_title: str,
    panels: Sequence[Mapping[str, Any]],
    group_names: Mapping[int, str],
    plot_cfg: Mapping[str, Any],
) -> None:
    if not panels:
        raise ValueError("No panels to plot.")

    plt = require_matplotlib()
    n_panels = len(panels)
    max_cols = int(plot_cfg.get("max_cols", 3))
    cols = min(max_cols, n_panels)
    rows = int(np.ceil(n_panels / cols))
    width = float(plot_cfg.get("width_per_panel", 4.0)) * cols
    height = float(plot_cfg.get("height_per_row", 3.7)) * rows + 0.8

    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    axes_flat = axes.reshape(-1)

    colors = plot_cfg.get("colors") or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = plot_cfg.get("markers") or ["o", "s", "^", "D"]
    point_size = float(plot_cfg.get("point_size", 8))
    alpha = float(plot_cfg.get("alpha", 0.72))

    group_order = [int(item) for item in plot_cfg.get("group_order", sorted(group_names))]
    for panel_idx, panel in enumerate(panels):
        ax = axes_flat[panel_idx]
        coords = np.asarray(panel["coords"])
        panel_labels = np.asarray(panel["labels"]).astype(int)

        for group_idx, group in enumerate(group_order):
            mask = panel_labels == group
            if not mask.any():
                continue
            scatter = ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                c=colors[group_idx % len(colors)],
                marker=markers[group_idx % len(markers)],
                alpha=alpha,
                linewidths=0,
                label=group_names.get(group, str(group)),
            )

        caption = f"({chr(ord('a') + panel_idx)}) {panel['method']}"
        ax.set_title(caption, fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(labelsize=8)
        ax.grid(False)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Domain adaptation t-SNE: Source-Learned Representations on {dataset_title} (target domain)",
        fontsize=12,
    )
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[idx % len(markers)],
            color="none",
            markerfacecolor=colors[idx % len(colors)],
            markeredgecolor="none",
            markersize=6,
            label=group_names.get(group, str(group)),
        )
        for idx, group in enumerate(group_order)
    ]
    labels = [group_names.get(group, str(group)) for group in group_order]
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            frameon=False,
            fontsize=9,
        )
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(plot_cfg.get("dpi", 300)), bbox_inches="tight")
    plt.close(fig)
