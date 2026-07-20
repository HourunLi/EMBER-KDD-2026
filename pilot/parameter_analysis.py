#!/usr/bin/env python3
"""Plot SFFGNN parameter-analysis results in the style used by CELL.

The original scripts in ``pilot/`` demonstrate several paper-figure styles,
but keep their data and output paths hard-coded.  This module retains those
visual conventions while providing a reusable command-line interface:

* one-dimensional sweeps reproduce ``topk_Bail.pdf``;
* ``tau_eta`` reproduces ``threshold_bailA_dp.pdf`` and writes DP/EO as
  separate PDFs;
* ``beta_gamma`` reproduces ``bail_dp.pdf`` and likewise writes DP/EO as
  separate PDFs.

The input is the aggregate ``summary.csv`` written by
``SFFGNN/parameter/run.py``.  All generated figures are PDF files with
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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


GROUPS = (
    "delta",
    "tau_eta",
    "beta_gamma",
    "lambda_residual_l2",
    "adapt_epochs",
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


# Exact colors recovered from the three CELL reference PDFs.
CELL_LINE_STYLES = (
    ("target_after_dp", "Demographic Parity", "#ADD8E6", "s", "-", 10),
    ("target_after_eo", "Equal Odds", "#FFD700", "*", "-", 13),
    ("target_after_acc", "Accuracy", "#90EE90", "v", "--", 12),
    ("target_after_auc", "ROC-AUC", "#FFDAB9", "o", "--", 12),
)

CELL_THRESHOLD_COLORS = (
    "#8491B4",
    "#91D1C2",
    "#3C5488",
    "#00A087",
    "#4DBBD5",
    "#8FBC8F",
    # EMBER has seven eta values while CELL has six threshold values.  The
    # additional muted purple extends the same low-saturation paper palette.
    "#B7A6C7",
)

# Front (small beta) to back (large beta), matching bail_dp.pdf exactly.
CELL_3D_ROW_COLORS = (
    "#EBB789",
    "#F8DFA8",
    "#E9F2F3",
    "#9BBEDE",
    "#5473AC",
)


def configure_style() -> None:
    """Apply the paper-style defaults shared by the legacy pilot scripts."""

    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.labelsize": 26,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 15,
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
    """Read an aggregate parameter-analysis CSV."""

    with path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for column in NUMERIC_COLUMNS:
            if column in row:
                row[column] = _as_number(row[column])
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


def _cell_line_ticks(axis, values: Sequence[float]) -> None:
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
    """Draw CELL-style utility/fairness curves for a one-dimensional sweep."""

    x_values = _sorted_values(rows, x_key)
    if not x_values:
        return
    positions = np.asarray(x_values, dtype=float)

    # topk_Bail.pdf was created from a 5 x 4 inch canvas and cropped tightly.
    fig, fairness_ax = plt.subplots(figsize=(5, 4))
    utility_ax = fairness_ax.twinx()

    handles = []
    labels = []
    fairness_values = []
    utility_values = []
    for series_index, (prefix, label, color, marker, linestyle, marker_size) in enumerate(
        CELL_LINE_STYLES
    ):
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
            linewidth=1.5,
            markersize=marker_size,
            markeredgecolor="black",
            markeredgewidth=0.05,
            label=label,
            zorder=3,
        )
        handles.append(handle)
        labels.append(label)

    fairness_ax.set_xlabel(PARAMETER_LABELS[x_key], fontsize=22)
    fairness_ax.set_ylabel("DP/EO", fontsize=22)
    utility_ax.set_ylabel("Acc/ROC-AUC", fontsize=22)
    fairness_ax.set_xticks(positions)
    fairness_ax.set_xticklabels([_format_value(value) for value in x_values])
    if len(x_values) > 1:
        x_span = x_values[-1] - x_values[0]
        fairness_ax.set_xlim(x_values[0] - 0.05 * x_span, x_values[-1] + 0.05 * x_span)
    fairness_ax.tick_params(axis="x", labelsize=15)
    fairness_ax.tick_params(axis="y", labelsize=20)
    utility_ax.tick_params(axis="y", labelsize=20)
    fairness_ax.grid(
        color="darkgrey",
        linestyle="--",
        axis="both",
        linewidth=0.8,
        alpha=0.3,
    )
    _cell_line_ticks(fairness_ax, fairness_values)
    _cell_line_ticks(utility_ax, utility_values)
    # ``twinx`` creates ``utility_ax`` after ``fairness_ax``.  An axes-level
    # legend attached to the left axis can therefore be overpainted by right-
    # axis curves even when the legend itself has a high local z-order.  Attach
    # the merged legend to the later axis and raise it above every line artist.
    legend = utility_ax.legend(
        handles,
        labels,
        loc="lower right",
        ncol=1,
        frameon=True,
        framealpha=0.8,
        prop={"family": "Times New Roman", "size": 10},
    )
    legend.set_zorder(1000)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
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
    """Reproduce CELL's one-metric threshold grouped-bar PDF."""

    x_values = _sorted_values(rows, x_key)
    series_values = _sorted_values(rows, series_key)
    # The CELL threshold panel contains six legend entries.  EMBER's seventh
    # point is eta=1 (the full Bayesian boundary); it remains in CSV/JSON but
    # is omitted from this paper panel to preserve the supplied six-bar layout.
    if len(series_values) > 6:
        series_values = series_values[:6]
    if not x_values or not series_values:
        return
    lookup = _row_lookup(rows, (x_key, series_key))
    positions = np.arange(len(x_values), dtype=float)
    epsilon = 0.02
    total_width = 0.82
    bar_width = (
        total_width - epsilon * max(0, len(series_values) - 1)
    ) / max(1, len(series_values))
    colors = [
        CELL_THRESHOLD_COLORS[index % len(CELL_THRESHOLD_COLORS)]
        for index in range(len(series_values))
    ]

    # threshold_bailA_dp.pdf uses an uncropped 5 x 4 inch canvas.
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
            alpha=0.8,
            edgecolor="none",
            label=rf"$\eta$={_format_value(series_value)}",
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
    axis.tick_params(axis="both", labelsize=25)
    axis.set_axisbelow(True)
    axis.grid(color="darkgrey", linestyle="-", axis="y", alpha=0.3)
    limits = _padded_limits(all_values, minimum_padding=0.1)
    if limits is not None:
        axis.set_ylim(*limits)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.legend(
        prop={"family": "Times New Roman", "size": 15},
        ncol=2,
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_3d_metric(
    axis,
    matrix: np.ndarray,
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_key: str,
    y_key: str,
    z_label: str,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    low = float(finite.min())
    high = float(finite.max())
    data_span = high - low

    # Keep the tallest bar visually prominent, as in CELL.  A fixed +/-0.5
    # margin makes narrow sweeps (for example Bail) look artificially flat.
    # Use a human-readable tick step, place the baseline just below the lowest
    # value, and reserve roughly 12% of the vertical range above the maximum.
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

    # CELL's legacy code submits all bars in one call.  Besides reproducing
    # its front-to-back color distribution, this lets mplot3d depth-sort every
    # face globally.  Small beta values are warm in front and large beta
    # values transition through white to blue toward the back.
    x_positions, y_positions = np.meshgrid(
        np.arange(len(x_values), dtype=float),
        np.arange(len(y_values), dtype=float),
        copy=False,
    )
    flat_values = matrix.ravel()
    valid = np.isfinite(flat_values)
    flat_y = y_positions.ravel()[valid]
    colors = [
        CELL_3D_ROW_COLORS[int(y_index) % len(CELL_3D_ROW_COLORS)]
        for y_index in flat_y
    ]
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
    short_labels = {
        "lambda_coord": r"$\gamma$",
        "lambda_fair": r"$\beta$",
        "proto_temp": r"$\tau$",
        "lambda_pi": r"$\eta$",
    }
    axis.xaxis.set_rotate_label(False)
    axis.yaxis.set_rotate_label(False)
    axis.zaxis.set_rotate_label(False)
    axis.set_xlabel(
        short_labels.get(x_key, PARAMETER_LABELS[x_key]),
        fontsize=50,
        rotation=0,
        labelpad=25,
    )
    axis.set_ylabel(
        short_labels.get(y_key, PARAMETER_LABELS[y_key]),
        fontsize=50,
        rotation=0,
        labelpad=25,
    )
    axis.set_zlabel(z_label, fontsize=50, rotation=90, labelpad=30)
    axis.set_zlim(bottom, upper)
    tick_count = int(math.floor((upper - bottom) / tick_step + 1e-9))
    axis.set_zticks(bottom + tick_step * np.arange(tick_count + 1))
    decimals = max(0, int(-math.floor(math.log10(tick_step))))
    axis.zaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    # Leave enough space between the large tick numerals and projected axis /
    # grid lines.  The legacy CELL padding is too tight at the larger paper
    # font sizes used here, especially after raising the camera slightly.
    axis.tick_params(axis="x", pad=10, labelsize=45)
    axis.tick_params(axis="y", pad=10, labelsize=45)
    axis.tick_params(axis="z", pad=13, labelsize=45)
    # Match CELL's projection: the smallest gamma and beta meet at the front
    # corner; gamma recedes to the right, beta recedes to the left, and the z
    # axis stays on the left.
    axis.view_init(elev=30, azim=-145)
    axis.set_box_aspect((1, 1, 0.75))


def plot_3d_fairness_bars(
    rows: Sequence[Mapping[str, object]],
    dataset: str,
    x_key: str,
    y_key: str,
    metric: str,
    metric_label: str,
    output_path: Path,
) -> None:
    """Reproduce CELL's one-metric 3D loss-weight PDF."""

    x_values = _sorted_values(rows, x_key)
    y_values = _sorted_values(rows, y_key)
    if not x_values or not y_values:
        return
    matrix = _pair_matrix(
        rows, x_key, y_key, x_values, y_values, f"{metric}_mean"
    )

    # The reference CropBox is approximately 12 x 10.5 inches.  An explicit
    # axes rectangle avoids Matplotlib's 3D tight-bbox bug clipping the z label.
    fig = plt.figure(figsize=(12, 10.5))
    finite = matrix[np.isfinite(matrix)]
    wide_z_range = bool(finite.size and float(finite.max() - finite.min()) > 2.0)
    axes_rect = (0.18, 0.16, 0.68, 0.72) if wide_z_range else (0.14, 0.07, 0.81, 0.88)
    axis = fig.add_axes(axes_rect, projection="3d")
    _plot_3d_metric(
        axis,
        matrix,
        x_values,
        y_values,
        x_key,
        y_key,
        f"{DATASET_LABELS.get(dataset, dataset)} {metric_label}",
    )
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
    image = axis.imshow(matrix, cmap="RdYlBu", aspect="auto", origin="lower")
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

            if group == "delta":
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
            elif group == "beta_gamma":
                dp_path = output_dir / f"beta_gamma_{dataset}_dp.pdf"
                eo_path = output_dir / f"beta_gamma_{dataset}_eo.pdf"
                plot_3d_fairness_bars(
                    group_rows,
                    dataset,
                    "lambda_coord",
                    "lambda_fair",
                    "target_after_dp",
                    "DP",
                    dp_path,
                )
                plot_3d_fairness_bars(
                    group_rows,
                    dataset,
                    "lambda_coord",
                    "lambda_fair",
                    "target_after_eo",
                    "EO",
                    eo_path,
                )
                generated.extend((dp_path, eo_path))

    return [path for path in generated if path.exists()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot SFFGNN parameter-analysis summary data."
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
