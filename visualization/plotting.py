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
    plt.rcParams.update(
        {
            "font.family": plot_cfg.get("font_family", "serif"),
            "font.serif": plot_cfg.get("font_serif", ["Times New Roman", "DejaVu Serif"]),
            "axes.unicode_minus": False,
        }
    )

    panel = panels[0]
    coords = np.asarray(panel["coords"])
    panel_labels = np.asarray(panel["labels"]).astype(int)

    fig_width = float(plot_cfg.get("figure_width", 5.35))
    fig_height = float(plot_cfg.get("figure_height", 4.35))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    colors = plot_cfg.get("colors") or ["#5B8DE8", "#FF7043", "#86B889", "#FFB6C8"]
    point_size = float(plot_cfg.get("point_size", 34))
    alpha = float(plot_cfg.get("alpha", 1.0))
    group_order = [int(item) for item in plot_cfg.get("group_order", sorted(group_names))]

    for group_idx, group in enumerate(group_order):
        mask = panel_labels == group
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            c=colors[group_idx % len(colors)],
            marker="o",
            alpha=alpha,
            linewidths=0,
        )

    margin = float(plot_cfg.get("axis_margin", 0.05))
    ax.set_xlim(0.0 - margin, 1.0 + margin)
    ax.set_ylim(0.0 - margin, 1.0 + margin)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    from matplotlib.ticker import FormatStrFormatter

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    tick_label_size = int(plot_cfg.get("tick_label_size", 28))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_label_size,
        length=float(plot_cfg.get("tick_length", 5.0)),
        width=float(plot_cfg.get("tick_width", 1.1)),
        direction="out",
        pad=float(plot_cfg.get("tick_pad", 5.0)),
    )

    spine_color = plot_cfg.get("spine_color", "#7F7F7F")
    spine_width = float(plot_cfg.get("spine_width", 1.3))
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(spine_width)

    xlabel_template = plot_cfg.get("xlabel_template", "{method} {dataset} t-SNE result")
    xlabel = xlabel_template.format(
        method=panel.get("method", ""),
        dataset=panel.get("dataset", dataset_title),
        dataset_title=dataset_title,
    )
    ax.set_xlabel(
        xlabel,
        fontsize=int(plot_cfg.get("axis_label_size", 34)),
        labelpad=float(plot_cfg.get("axis_label_pad", 6.0)),
    )
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)

    if plot_cfg.get("show_legend", False):
        ax.legend(frameon=False, fontsize=int(plot_cfg.get("legend_size", 12)))

    fig.tight_layout(pad=float(plot_cfg.get("tight_layout_pad", 0.2)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(plot_cfg.get("dpi", 300)), bbox_inches="tight")
    plt.close(fig)
