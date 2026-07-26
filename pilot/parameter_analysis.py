#!/usr/bin/env python3
"""Plot EMBER parameter-analysis results with a polished publication style.

The figures use the supplied scientific colour references, compact typography,
subtle grids, and consistent two-dimensional heatmaps for the three paper
parameter groups.

The input is the aggregate ``summary.csv`` written by
``EMBER/parameter/run.py``.  All generated figures are PDF files with
embedded TrueType fonts.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.transforms import Bbox


PAPER_GROUPS = ("beta_gamma", "nu0_delta", "eta_omega")
SUPPLEMENTARY_GROUPS = (
    "delta",
    "tau_eta",
    "lambda_residual_l2",
    "adapt_epochs",
)
GROUPS = PAPER_GROUPS + SUPPLEMENTARY_GROUPS

PAIR_PARAMETERS = {
    "beta_gamma": ("lambda_coord", "lambda_fair"),
    "nu0_delta": ("tau_c", "group_pseudocount"),
    "eta_omega": ("prior_discount", "lambda_pi"),
}

HEATMAP_METRICS = (
    ("target_after_acc", "acc", "Accuracy", False),
    ("target_after_auc", "auc", "ROC-AUC", False),
    ("target_after_dp", "dp", "DP", True),
    ("target_after_eo", "eo", "EO", True),
)

DATASET_LABELS = {
    "bailA": "Bail",
    "germanA": "German",
    "pokec": "Pokec",
    "syn": "Syn",
}

PARAMETER_LABELS = {
    "tau_c": r"Confidence threshold $\delta$",
    "proto_temp": r"Prototype temperature $\tau$",
    "lambda_pi": r"Prior strength $\eta$",
    "prior_discount": r"Prior discount $\omega$",
    "group_pseudocount": r"Group pseudocount $\nu_0$",
    "lambda_fair": r"Fairness weight $\beta$",
    "lambda_coord": r"Coordination weight $\gamma$",
    "lambda_residual_l2": r"Residual regularization $\lambda$",
    "adapt_epochs": r"Adaptation steps $T$",
}

NUMERIC_COLUMNS = {
    "runs_completed",
    "tau_c",
    "proto_temp",
    "lambda_pi",
    "prior_discount",
    "group_pseudocount",
    "lambda_fair",
    "lambda_coord",
    "lambda_residual_l2",
    "adapt_epochs",
}
for _stage in ("source", "target_before", "target_after"):
    for _metric in ("acc", "auc", "dp", "eo"):
        NUMERIC_COLUMNS.add(f"{_stage}_{_metric}_mean")
        NUMERIC_COLUMNS.add(f"{_stage}_{_metric}_std")
NUMERIC_COLUMNS.update({"composite_mean", "composite_std"})


# Unified low-saturation "sea breeze" scientific palette.
DUSK_COLORS = (
    "#51999F",
    "#4198AC",
    "#7BCCCD",
    "#BFDFD2",
    "#DBCB92",
    "#ECB66C",
    "#ED8D5A",
)

LINE_STYLES = (
    ("target_after_dp", "DP", "#67B7C7", "D", "-"),
    ("target_after_eo", "EO", "#8FD3C8", "s", "-"),
    ("target_after_acc", "Accuracy", "#F2D27A", "o", "--"),
    ("target_after_auc", "ROC-AUC", "#F39A6B", "v", "--"),
)

SEA_BREEZE_CMAP = LinearSegmentedColormap.from_list(
    "sea_breeze",
    (
        "#4198AC",
        "#51999F",
        "#7BCCCD",
        "#BFDFD2",
        "#DBCB92",
        "#ECB66C",
        "#EA9E58",
        "#ED8D5A",
    ),
)

BETA_GAMMA_3D_COLORS = ("#51999F", "#7BCCCD", "#BFDFD2", "#DBCB92", "#ECB66C")


def configure_style() -> None:
    """Apply restrained serif defaults for all parameter figures."""

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
            "savefig.facecolor": "white",
        }
    )


def _as_number(value: str):
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_summary(path: Path) -> List[Dict[str, object]]:
    """Read a paper-aligned aggregate parameter-analysis CSV."""

    with path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for column in NUMERIC_COLUMNS:
            if column in row:
                row[column] = _as_number(row[column])
    invalid_gamma_rows = [
        row
        for row in rows
        if row.get("group") == "beta_gamma"
        and _finite(row.get("lambda_coord"))
        and not 0.0 <= float(row["lambda_coord"]) <= 1.0
    ]
    if invalid_gamma_rows:
        raise ValueError(
            "Summary contains beta-gamma rows outside paper Section 3.3's "
            "gamma range [0, 1]. Re-aggregate the raw results with "
            "EMBER/parameter/run.py before plotting."
        )
    invalid_nu0_rows = [
        row
        for row in rows
        if row.get("group") == "nu0_delta"
        and _finite(row.get("group_pseudocount"))
        and float(row["group_pseudocount"]) <= 0.0
    ]
    if invalid_nu0_rows:
        raise ValueError("Summary contains nu0-delta rows with non-positive nu_0.")
    invalid_omega_rows = [
        row
        for row in rows
        if row.get("group") == "eta_omega"
        and _finite(row.get("prior_discount"))
        and not 0.0 <= float(row["prior_discount"]) < 1.0
    ]
    if invalid_omega_rows:
        raise ValueError("Summary contains eta-omega rows with omega outside [0, 1).")
    return rows


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_value(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:g}"


def _sorted_values(rows: Sequence[Mapping[str, object]], key: str) -> List[float]:
    return sorted({float(row[key]) for row in rows if _finite(row.get(key))})


def _row_lookup(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> Dict[tuple, Mapping[str, object]]:
    lookup = {}
    for row in rows:
        if all(_finite(row.get(key)) for key in keys):
            lookup[tuple(float(row[key]) for key in keys)] = row
    return lookup


def _metric_array(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    x_values: Sequence[float],
    metric: str,
) -> np.ndarray:
    lookup = _row_lookup(rows, (x_key,))
    return np.asarray(
        [float(lookup[(value,)][metric]) if (value,) in lookup else np.nan for value in x_values],
        dtype=float,
    )


def _row_major_legend_entries(axis, ncol: int):
    """Reorder handles so a multi-row Matplotlib legend reads left to right."""

    handles, labels = axis.get_legend_handles_labels()
    nrows = int(math.ceil(len(handles) / ncol))
    order = [
        row * ncol + column
        for column in range(ncol)
        for row in range(nrows)
        if row * ncol + column < len(handles)
    ]
    return [handles[index] for index in order], [labels[index] for index in order]


def _padded_limits(values: Sequence[float], minimum_padding: float = 0.1):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    padding = max(minimum_padding, span * 0.15)
    lower = low - padding
    upper = high + padding
    if low >= 0 and lower < 0:
        lower = 0.0
    if math.isclose(lower, upper):
        upper = lower + max(1.0, abs(lower) * 0.1)
    return lower, upper


def _line_ticks(axis, values: Sequence[float]) -> None:
    limits = _padded_limits(values, minimum_padding=0.25)
    if limits is not None:
        axis.set_ylim(*limits)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    span = float(finite.max() - finite.min()) if finite.size else 1.0
    axis.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f" if span < 3.0 else "%.0f")
    )


def plot_single_parameter(
    rows: Sequence[Mapping[str, object]],
    dataset: str,
    group: str,
    x_key: str,
    output_path: Path,
) -> None:
    """Draw the legacy framed utility/fairness curves for a one-dimensional sweep."""

    x_values = _sorted_values(rows, x_key)
    if not x_values:
        return
    positions = np.asarray(x_values, dtype=float)

    # Use a fixed canvas and fixed axes rectangle so datasets with wider tick
    # labels do not produce differently cropped/scaled data frames.
    fig, fairness_ax = plt.subplots(figsize=(6, 4))
    utility_ax = fairness_ax.twinx()

    handles = []
    labels = []
    fairness_values = []
    utility_values = []
    for series_index, (prefix, label, color, marker, linestyle) in enumerate(LINE_STYLES):
        axis = fairness_ax if series_index < 2 else utility_ax
        means = _metric_array(rows, x_key, x_values, f"{prefix}_mean")
        if series_index < 2:
            fairness_values.extend(means.tolist())
        else:
            utility_values.extend(means.tolist())
        handle, = axis.plot(
            positions,
            means,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=10.5,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=label,
            zorder=3,
        )
        handles.append(handle)
        labels.append(label)

    fairness_ax.set_xlabel(PARAMETER_LABELS[x_key], fontsize=24)
    fairness_ax.set_ylabel("DP/EO", fontsize=24)
    utility_ax.set_ylabel("Acc/ROC-AUC", fontsize=24)
    fairness_ax.set_xticks(positions)
    fairness_ax.set_xticklabels([_format_value(value) for value in x_values])
    if len(x_values) > 1:
        x_span = x_values[-1] - x_values[0]
        fairness_ax.set_xlim(
            x_values[0] - 0.05 * x_span,
            x_values[-1] + 0.05 * x_span,
        )
    fairness_ax.tick_params(axis="x", labelsize=16, colors="#333333")
    fairness_ax.tick_params(axis="y", labelsize=22, colors="#333333")
    utility_ax.tick_params(axis="y", labelsize=22, colors="#333333")
    fairness_ax.grid(
        color="darkgrey",
        linestyle="--",
        axis="both",
        linewidth=0.8,
        alpha=0.3,
    )
    fairness_ax.set_axisbelow(True)
    for spine in fairness_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    utility_ax.spines["top"].set_visible(False)
    utility_ax.spines["bottom"].set_visible(False)
    utility_ax.spines["left"].set_visible(False)
    utility_ax.spines["right"].set_visible(True)
    utility_ax.spines["right"].set_color("black")
    _line_ticks(fairness_ax, fairness_values)
    _line_ticks(utility_ax, utility_values)
    legend = fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.15, 0.845, 0.60, 0.10),
        mode="expand",
        ncol=4,
        frameon=False,
        fontsize=15,
        borderaxespad=0.0,
        borderpad=0.0,
        columnspacing=0.25,
        handlelength=0.85,
        handletextpad=0.25,
    )
    legend.set_zorder(1000)
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.15, right=0.75)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _pair_matrix(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    y_key: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    metric: str,
) -> np.ndarray:
    lookup = _row_lookup(rows, (x_key, y_key))
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            row = lookup.get((x_value, y_value))
            if row is not None and _finite(row.get(metric)):
                matrix[y_index, x_index] = float(row[metric])
    return matrix


def plot_grouped_fairness_bars(
    rows: Sequence[Mapping[str, object]],
    dataset: str,
    x_key: str,
    series_key: str,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Draw a polished grouped-bar chart for a two-parameter sweep."""

    x_values = _sorted_values(rows, x_key)
    series_values = _sorted_values(rows, series_key)
    if not x_values or not series_values:
        return
    lookup = _row_lookup(rows, (x_key, series_key))
    positions = np.arange(len(x_values), dtype=float)
    epsilon = 0.012
    total_width = 0.86
    bar_width = (
        total_width - epsilon * max(0, len(series_values) - 1)
    ) / max(1, len(series_values))
    colors = [
        DUSK_COLORS[index % len(DUSK_COLORS)]
        for index in range(len(series_values))
    ]

    fig, axis = plt.subplots(figsize=(5, 4))
    all_values = []
    for series_index, series_value in enumerate(series_values):
        means = []
        for x_value in x_values:
            row = lookup.get((x_value, series_value))
            means.append(
                float(row[f"{metric}_mean"])
                if row is not None and _finite(row.get(f"{metric}_mean"))
                else np.nan
            )
        all_values.extend(means)
        offset = series_index * (bar_width + epsilon)
        axis.bar(
            positions + offset,
            means,
            width=bar_width,
            color=colors[series_index],
            alpha=1.0,
            edgecolor="white",
            linewidth=0.6,
            label=_format_value(series_value),
            zorder=3,
        )

    group_center = 0.5 * (
        (len(series_values) - 1) * (bar_width + epsilon) + bar_width
    )
    axis.set_xlabel(PARAMETER_LABELS[x_key], fontsize=26)
    axis.set_ylabel(ylabel, fontsize=26)
    axis.set_xticks(positions + group_center)
    axis.set_xticklabels([_format_value(value) for value in x_values])
    axis.set_xlim(-0.15, len(x_values) - 1 + total_width + 0.15)
    axis.tick_params(axis="both", colors="#333333", labelsize=25)
    axis.set_axisbelow(True)
    axis.grid(color="#E4E6EA", linestyle="-", axis="y", linewidth=0.7, alpha=0.9)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    limits = _padded_limits(all_values, minimum_padding=0.1)
    if limits is not None:
        axis.set_ylim(*limits)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    legend_columns = len(series_values)
    legend_handles, legend_labels = _row_major_legend_entries(axis, legend_columns)
    legend = axis.legend(
        legend_handles,
        legend_labels,
        title=rf"Prior strength $\eta$",
        ncol=legend_columns,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.16),
        mode="expand",
        frameon=False,
        fontsize=18,
        borderaxespad=0.0,
        handlelength=0.60,
        handleheight=0.8,
        handletextpad=0,
        columnspacing=0.15,
        borderpad=0.0,
    )
    legend.get_title().set_fontsize(20)
    fig.subplots_adjust(top=0.72, bottom=0.22, left=0.17, right=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_3d_fairness_bars(
    rows: Sequence[Mapping[str, object]],
    dataset: str,
    x_key: str,
    y_key: str,
    metric: str,
    metric_label: str,
    output_path: Path,
) -> None:
    """Draw the beta-gamma grid with the legacy 3D geometry and typography."""

    x_values = _sorted_values(rows, x_key)
    y_values = _sorted_values(rows, y_key)
    if not x_values or not y_values:
        return
    matrix = _pair_matrix(
        rows, x_key, y_key, x_values, y_values, f"{metric}_mean"
    )
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return

    low = float(finite.min())
    high = float(finite.max())
    data_span = high - low
    scale_span = max(data_span, max(abs(low), abs(high), 1.0) * 0.02)
    rough_step = scale_span / 3.0
    exponent = math.floor(math.log10(rough_step))
    magnitude = 10.0**exponent
    fraction = rough_step / magnitude
    if fraction <= 1.0:
        tick_step = magnitude
    elif fraction <= 2.0:
        tick_step = 2.0 * magnitude
    elif fraction <= 5.0:
        tick_step = 5.0 * magnitude
    else:
        tick_step = 10.0 * magnitude

    bottom = math.floor((low - 0.2 * scale_span) / tick_step) * tick_step
    if low >= 0.0:
        bottom = max(0.0, bottom)
    bar_height = high - bottom
    if bar_height <= 0.0:
        bottom = high - tick_step
        if high >= 0.0:
            bottom = max(0.0, bottom)
        bar_height = max(high - bottom, tick_step)
    upper = bottom + bar_height / 0.88

    x_positions, y_positions = np.meshgrid(
        np.arange(len(x_values), dtype=float),
        np.arange(len(y_values), dtype=float),
        copy=False,
    )
    flat_values = matrix.ravel()
    valid = np.isfinite(flat_values)
    flat_y = y_positions.ravel()[valid]
    colors = [
        BETA_GAMMA_3D_COLORS[int(y_index) % len(BETA_GAMMA_3D_COLORS)]
        for y_index in flat_y
    ]

    fig = plt.figure(figsize=(12, 10.5))
    # Keep the projected axes at exactly the same physical size for every
    # dataset.  German previously used a smaller rectangle for its wider
    # z-range, which made that PDF look smaller after tight cropping.
    axes_rect = (0.14, 0.07, 0.81, 0.88)
    axis = fig.add_axes(axes_rect, projection="3d")
    axis.bar3d(
        x_positions.ravel()[valid],
        flat_y,
        np.full(valid.sum(), bottom),
        np.full(valid.sum(), 0.5),
        np.full(valid.sum(), 0.5),
        flat_values[valid] - bottom,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.05,
        shade=False,
    )

    axis.set_xticks(np.arange(len(x_values)) + 0.25)
    axis.set_xticklabels([_format_value(value) for value in x_values])
    axis.set_yticks(np.arange(len(y_values)) + 0.25)
    axis.set_yticklabels([_format_value(value) for value in y_values])
    axis.xaxis.set_rotate_label(False)
    axis.yaxis.set_rotate_label(False)
    axis.zaxis.set_rotate_label(False)
    axis.set_xlabel(r"$\gamma$", fontsize=50, rotation=0, labelpad=25)
    axis.set_ylabel(r"$\beta$", fontsize=50, rotation=0, labelpad=25)
    axis.set_zlabel(
        f"{DATASET_LABELS.get(dataset, dataset)} {metric_label}",
        fontsize=50,
        rotation=90,
        labelpad=30,
    )
    axis.set_zlim(bottom, upper)
    tick_count = int(math.floor((upper - bottom) / tick_step + 1e-9))
    axis.set_zticks(bottom + tick_step * np.arange(tick_count + 1))
    decimals = max(0, int(-math.floor(math.log10(tick_step))))
    axis.zaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    axis.tick_params(axis="x", pad=10, labelsize=45)
    axis.tick_params(axis="y", pad=10, labelsize=45)
    axis.tick_params(axis="z", pad=13, labelsize=45)
    axis.view_init(elev=30, azim=-145)
    axis.set_box_aspect((1, 1, 0.75))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the complete coloured 3D pane at the top, while bringing the
    # left, right, and bottom edges close to their outermost visible content.
    # The shared CropBox keeps all eight PDFs identically sized.
    output_bbox = Bbox.from_bounds(0.85, 0.58, 10.10, 8.87)
    fig.savefig(output_path, bbox_inches=output_bbox, pad_inches=0.0)
    plt.close(fig)


def plot_parameter_heatmap(
    rows: Sequence[Mapping[str, object]],
    dataset: str,
    x_key: str,
    y_key: str,
    metric: str,
    metric_label: str,
    lower_is_better: bool,
    output_path: Path,
) -> None:
    """Draw a readable two-dimensional map of a parameter grid."""

    x_values = _sorted_values(rows, x_key)
    y_values = _sorted_values(rows, y_key)
    if not x_values or not y_values:
        return
    matrix = _pair_matrix(
        rows, x_key, y_key, x_values, y_values, f"{metric}_mean"
    )
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return

    fig, axis = plt.subplots(figsize=(12, 10.5))
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(
        masked,
        cmap=SEA_BREEZE_CMAP,
        aspect="equal",
        origin="lower",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(x_values)))
    axis.set_xticklabels([_format_value(value) for value in x_values])
    axis.set_yticks(np.arange(len(y_values)))
    axis.set_yticklabels([_format_value(value) for value in y_values])
    axis.set_xlabel(PARAMETER_LABELS[x_key], fontsize=50)
    axis.set_ylabel(PARAMETER_LABELS[y_key], fontsize=50)
    axis.set_title(
        f"{DATASET_LABELS.get(dataset, dataset)} {metric_label}",
        fontsize=50,
    )
    axis.tick_params(length=0, labelsize=45)
    axis.set_xticks(np.arange(-0.5, len(x_values), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(y_values), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    axis.tick_params(which="minor", bottom=False, left=False)

    low = float(finite.min())
    high = float(finite.max())
    span = max(high - low, 1e-12)
    for y_index in range(len(y_values)):
        for x_index in range(len(x_values)):
            value = matrix[y_index, x_index]
            if not np.isfinite(value):
                continue
            relative = (float(value) - low) / span
            axis.text(
                x_index,
                y_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=45,
                color="white" if relative >= 0.55 else "#26334A",
            )

    colorbar = fig.colorbar(image, ax=axis, fraction=0.047, pad=0.035)
    direction = "lower is better" if lower_is_better else "higher is better"
    colorbar.set_label(f"{metric_label} ({direction})", fontsize=50)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=45, length=2)
    fig.tight_layout(pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_composite_heatmap(
    rows: Sequence[Mapping[str, object]],
    dataset: str,
    group: str,
    x_key: str,
    y_key: str,
    output_path: Path,
) -> None:
    """Draw an annotated utility-fairness composite-score heatmap."""

    x_values = _sorted_values(rows, x_key)
    y_values = _sorted_values(rows, y_key)
    if not x_values or not y_values:
        return
    matrix = _pair_matrix(rows, x_key, y_key, x_values, y_values, "composite_mean")
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return

    fig, axis = plt.subplots(figsize=(4.7, 3.55))
    image = axis.imshow(matrix, cmap=SEA_BREEZE_CMAP, aspect="auto", origin="lower")
    axis.set_xticks(np.arange(len(x_values)))
    axis.set_xticklabels([_format_value(value) for value in x_values])
    axis.set_yticks(np.arange(len(y_values)))
    axis.set_yticklabels([_format_value(value) for value in y_values])
    axis.set_xlabel(PARAMETER_LABELS[x_key])
    axis.set_ylabel(PARAMETER_LABELS[y_key])
    axis.set_title(
        f"{DATASET_LABELS.get(dataset, dataset)} composite score"
    )
    threshold = float(np.nanmedian(matrix))
    for y_index in range(len(y_values)):
        for x_index in range(len(x_values)):
            value = matrix[y_index, x_index]
            if not np.isfinite(value):
                continue
            axis.text(
                x_index,
                y_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < threshold else "black",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
    colorbar.set_label(r"ACC + AUC $-$ DP $-$ EO")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_figures(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    groups: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
) -> List[Path]:
    """Generate all applicable figures and return their paths."""

    configure_style()
    selected_groups = set(groups or GROUPS)
    available_datasets = sorted(
        {str(row["dataset"]) for row in rows if row.get("dataset")}
    )
    selected_datasets = set(datasets or available_datasets)
    generated: List[Path] = []

    for dataset in available_datasets:
        if dataset not in selected_datasets:
            continue
        for group in GROUPS:
            if group not in selected_groups:
                continue
            group_rows = [
                row
                for row in rows
                if row.get("dataset") == dataset and row.get("group") == group
            ]
            if not group_rows:
                continue

            if group in PAIR_PARAMETERS:
                x_key, y_key = PAIR_PARAMETERS[group]
                for metric, suffix, label, lower_is_better in HEATMAP_METRICS:
                    path = output_dir / f"{group}_{dataset}_{suffix}.pdf"
                    plot_parameter_heatmap(
                        group_rows,
                        dataset,
                        x_key,
                        y_key,
                        metric,
                        label,
                        lower_is_better,
                        path,
                    )
                    generated.append(path)
            elif group == "delta":
                path = output_dir / f"delta_{dataset}.pdf"
                plot_single_parameter(group_rows, dataset, group, "tau_c", path)
                generated.append(path)
            elif group == "lambda_residual_l2":
                path = output_dir / f"lambda_residual_l2_{dataset}.pdf"
                plot_single_parameter(
                    group_rows,
                    dataset,
                    group,
                    "lambda_residual_l2",
                    path,
                )
                generated.append(path)
            elif group == "adapt_epochs":
                path = output_dir / f"adapt_epochs_{dataset}.pdf"
                plot_single_parameter(
                    group_rows, dataset, group, "adapt_epochs", path
                )
                generated.append(path)
            elif group == "tau_eta":
                dp_path = output_dir / f"tau_eta_{dataset}_dp.pdf"
                eo_path = output_dir / f"tau_eta_{dataset}_eo.pdf"
                plot_grouped_fairness_bars(
                    group_rows,
                    dataset,
                    "proto_temp",
                    "lambda_pi",
                    "target_after_dp",
                    "Demographic Parity",
                    dp_path,
                )
                plot_grouped_fairness_bars(
                    group_rows,
                    dataset,
                    "proto_temp",
                    "lambda_pi",
                    "target_after_eo",
                    "Equal Odds",
                    eo_path,
                )
                generated.extend((dp_path, eo_path))
    return [path for path in generated if path.exists()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot EMBER parameter-analysis summary data."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=GROUPS)
    parser.add_argument("--datasets", nargs="+")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.summary.exists():
        raise FileNotFoundError(f"Summary file does not exist: {args.summary}")
    rows = load_summary(args.summary)
    paths = generate_figures(
        rows,
        args.output_dir,
        groups=args.groups,
        datasets=args.datasets,
    )
    if not paths:
        raise RuntimeError("No figures were generated from the supplied summary")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
