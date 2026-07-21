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
    spine_color = plot_cfg.get("spine_color", "#808080")
    spine_width = float(plot_cfg.get("spine_width", 1.0))
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": plot_cfg.get("font_family", "Times New Roman"),
            "font.serif": plot_cfg.get("font_serif", ["Times New Roman", "DejaVu Serif"]),
            "font.style": "normal",
            "font.weight": "normal",
            "axes.edgecolor": spine_color,
            "axes.linewidth": spine_width,
            "axes.unicode_minus": False,
        }
    )

    panel = panels[0]
    coords = np.asarray(panel["coords"])
    panel_labels = np.asarray(panel["labels"]).astype(int)

    fig_width = float(plot_cfg.get("figure_width", 6.4))
    fig_height = float(plot_cfg.get("figure_height", 4.8))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    colors = plot_cfg.get("colors") or ["#FF7F50", "#6495ED", "#8FBC8F", "#FFC0CB"]
    point_size = float(plot_cfg.get("point_size", 20))
    alpha = float(plot_cfg.get("alpha", 1.0))
    group_order = [int(item) for item in plot_cfg.get("group_order", sorted(group_names))]
    color_by_group = {
        group: colors[group_idx % len(colors)] for group_idx, group in enumerate(group_order)
    }
    unknown_groups = sorted(set(panel_labels) - set(color_by_group))
    if unknown_groups:
        raise ValueError(f"No plot color configured for groups: {unknown_groups}")

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=point_size,
        c=[color_by_group[label] for label in panel_labels],
        marker="o",
        alpha=alpha,
        linewidths=float(plot_cfg.get("marker_line_width", 0.3)),
    )

    margin = float(plot_cfg.get("axis_margin", 0.05))
    ax.set_xlim(0.0 - margin, 1.0 + margin)
    ax.set_ylim(0.0 - margin, 1.0 + margin)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    from matplotlib.ticker import FormatStrFormatter

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    tick_label_size = int(plot_cfg.get("tick_label_size", 18))
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_label_size,
        length=float(plot_cfg.get("tick_length", 3.5)),
        width=float(plot_cfg.get("tick_width", 0.8)),
        direction="out",
        pad=float(plot_cfg.get("tick_pad", 3.5)),
    )

    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(spine_width)

    xlabel_template = plot_cfg.get(
        "xlabel_template", "{method} {dataset_title} t-SNE result"
    )
    xlabel = xlabel_template.format(
        method=panel.get("method", ""),
        dataset=panel.get("dataset", dataset_title),
        dataset_title=dataset_title,
    )
    ax.set_xlabel(
        xlabel,
        fontsize=int(plot_cfg.get("axis_label_size", 20)),
        labelpad=float(plot_cfg.get("axis_label_pad", 3.4)),
    )
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)

    if plot_cfg.get("show_legend", False):
        ax.legend(frameon=False, fontsize=int(plot_cfg.get("legend_size", 12)))

    if plot_cfg.get("tight_layout", False):
        fig.tight_layout(pad=float(plot_cfg.get("tight_layout_pad", 0.2)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=int(plot_cfg.get("dpi", 300)),
        bbox_inches="tight",
        pad_inches=float(plot_cfg.get("savefig_pad_inches", 0.1)),
    )
    plt.close(fig)
