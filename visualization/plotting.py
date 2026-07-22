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
            "font.family": plot_cfg.get("font_family", "serif"),
            "font.serif": plot_cfg.get(
                "font_serif", ["Times New Roman", "Times", "DejaVu Serif"]
            ),
            "font.sans-serif": plot_cfg.get(
                "font_sans_serif", ["DejaVu Sans", "Arial", "Liberation Sans"]
            ),
            "font.style": "normal",
            "font.weight": "normal",
            "axes.edgecolor": spine_color,
            "axes.facecolor": plot_cfg.get("axes_facecolor", "#FAFAFA"),
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

    colors = plot_cfg.get("colors") or ["#003566", "#9E2A2B", "#669BBC", "#E29578"]
    markers = plot_cfg.get("markers") or ["o", "o", "o", "o"]
    point_size = float(plot_cfg.get("point_size", 11))
    alpha = float(plot_cfg.get("alpha", 0.70))
    group_order = [int(item) for item in plot_cfg.get("group_order", sorted(group_names))]
    color_by_group = {
        group: colors[group_idx % len(colors)] for group_idx, group in enumerate(group_order)
    }
    marker_by_group = {
        group: markers[group_idx % len(markers)] for group_idx, group in enumerate(group_order)
    }
    unknown_groups = sorted(set(panel_labels) - set(color_by_group))
    if unknown_groups:
        raise ValueError(f"No plot color configured for groups: {unknown_groups}")

    marker_line_width = float(plot_cfg.get("marker_line_width", 0.0))
    marker_edge_color = plot_cfg.get("marker_edge_color", "none")
    legend_handles = None
    if len(set(marker_by_group.values())) == 1:
        draw_order = np.random.default_rng(0).permutation(panel_labels.shape[0])
        ordered_labels = panel_labels[draw_order]
        ax.scatter(
            coords[draw_order, 0],
            coords[draw_order, 1],
            s=point_size,
            c=[color_by_group[label] for label in ordered_labels],
            marker=next(iter(marker_by_group.values())),
            alpha=alpha,
            linewidths=marker_line_width,
            edgecolors=marker_edge_color,
        )
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=marker_by_group[group],
                markersize=5.5,
                markerfacecolor=color_by_group[group],
                markeredgecolor="none",
                label=group_names.get(group, str(group)),
            )
            for group in group_order
            if np.any(panel_labels == group)
        ]
    else:
        for group in group_order:
            mask = panel_labels == group
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                color=color_by_group[group],
                marker=marker_by_group[group],
                alpha=alpha,
                linewidths=marker_line_width,
                edgecolors=marker_edge_color,
                label=group_names.get(group, str(group)),
            )

    margin = float(plot_cfg.get("axis_margin", 0.05))
    ax.set_xlim(0.0 - margin, 1.0 + margin)
    ax.set_ylim(0.0 - margin, 1.0 + margin)
    if plot_cfg.get("show_ticks", False):
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
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(spine_width)
        spine.set_visible(bool(plot_cfg.get("show_spines", False)))

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
    title_template = plot_cfg.get("title_template", "{method} on {dataset_title}")
    title = title_template.format(
        method=panel.get("method", ""),
        dataset=panel.get("dataset", dataset_title),
        dataset_title=dataset_title,
    )
    ax.set_title(
        title,
        fontsize=int(plot_cfg.get("title_size", 14)),
        pad=float(plot_cfg.get("title_pad", 8)),
    )
    ax.grid(False)
    if plot_cfg.get("equal_aspect", True):
        ax.set_aspect("equal", adjustable="box")

    if plot_cfg.get("show_legend", False):
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=int(plot_cfg.get("legend_ncol", 4)),
            frameon=False,
            fontsize=float(plot_cfg.get("legend_size", 9.5)),
            markerscale=float(plot_cfg.get("legend_marker_scale", 1.25)),
            handletextpad=0.35,
            columnspacing=1.1,
            borderaxespad=0.0,
        )

    if plot_cfg.get("tight_layout", False):
        fig.tight_layout(pad=float(plot_cfg.get("tight_layout_pad", 0.2)))
    elif plot_cfg.get("show_legend", False):
        fig.subplots_adjust(top=0.89, bottom=0.12, left=0.03, right=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=int(plot_cfg.get("dpi", 300)),
        bbox_inches="tight",
        pad_inches=float(plot_cfg.get("savefig_pad_inches", 0.1)),
    )
    plt.close(fig)
