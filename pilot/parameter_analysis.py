#!/usr/bin/env python3
"""Plot SFFGNN parameter-analysis results in the style used by CELL.

The original scripts in ``pilot/`` demonstrate several paper-figure styles,
but keep their data and output paths hard-coded.  This module retains those
visual conventions while providing a reusable command-line interface:

* one-dimensional sweeps use CELL-style utility/fairness line charts;
* ``tau_eta`` uses grouped fairness bars, matching CELL's threshold plot;
* ``beta_gamma`` uses paired 3D fairness bars, matching CELL's loss plot;
* every two-dimensional sweep also gets a composite-score heatmap for easier
  numerical comparison.

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
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter


GROUPS = (
    "delta",
    "tau_eta",
    "beta_gamma",
    "lambda_residual_l2",
    "adapt_epochs",
)

DATASET_LABELS = {
    "bailA": "BailA",
    "germanA": "GermanA",
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


def configure_style() -> None:
    """Apply the paper-style defaults shared by the legacy pilot scripts."""

    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#666666",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.bbox": "tight",
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
    positions = np.arange(len(x_values), dtype=float)

    fig, fairness_ax = plt.subplots(figsize=(4.45, 3.05))
    utility_ax = fairness_ax.twinx()

    definitions = (
        (fairness_ax, "target_after_dp", "DP", "#3C5488", "s"),
        (fairness_ax, "target_after_eo", "EO", "#00A087", "D"),
        (utility_ax, "target_after_acc", "Accuracy", "#E64B35", "o"),
        (utility_ax, "target_after_auc", "ROC-AUC", "#F39B7F", "^"),
    )
    handles = []
    labels = []
    for axis, prefix, label, color, marker in definitions:
        means = _metric_array(rows, x_key, x_values, f"{prefix}_mean")
        stds = _metric_array(rows, x_key, x_values, f"{prefix}_std")
        handle = axis.errorbar(
            positions,
            means,
            yerr=stds,
            color=color,
            marker=marker,
            linewidth=1.45,
            markersize=4.2,
            capsize=2.0,
            elinewidth=0.7,
            label=label,
            zorder=3,
        )
        handles.append(handle)
        labels.append(label)

    fairness_ax.set_xlabel(PARAMETER_LABELS[x_key])
    fairness_ax.set_ylabel("DP / EO (%)")
    utility_ax.set_ylabel("Accuracy / ROC-AUC (%)")
    fairness_ax.set_xticks(positions)
    fairness_ax.set_xticklabels([_format_value(value) for value in x_values])
    fairness_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    fairness_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    utility_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    fairness_ax.set_title(DATASET_LABELS.get(dataset, dataset))
    fairness_ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        columnspacing=0.9,
        handlelength=1.6,
    )
    fig.tight_layout()
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
    output_path: Path,
) -> None:
    """Draw the grouped threshold bars used by CELL for two parameters."""

    x_values = _sorted_values(rows, x_key)
    series_values = _sorted_values(rows, series_key)
    if not x_values or not series_values:
        return
    lookup = _row_lookup(rows, (x_key, series_key))
    positions = np.arange(len(x_values), dtype=float)
    total_width = 0.82
    bar_width = total_width / max(1, len(series_values))
    colors = cm.get_cmap("GnBu")(np.linspace(0.30, 0.92, len(series_values)))

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.85), sharex=True)
    for axis, metric, ylabel in (
        (axes[0], "target_after_dp", "Demographic Parity (%)"),
        (axes[1], "target_after_eo", "Equalized Odds (%)"),
    ):
        for series_index, series_value in enumerate(series_values):
            means = []
            stds = []
            for x_value in x_values:
                row = lookup.get((x_value, series_value))
                means.append(
                    float(row[f"{metric}_mean"])
                    if row is not None and _finite(row.get(f"{metric}_mean"))
                    else np.nan
                )
                stds.append(
                    float(row[f"{metric}_std"])
                    if row is not None and _finite(row.get(f"{metric}_std"))
                    else 0.0
                )
            offset = -total_width / 2 + (series_index + 0.5) * bar_width
            axis.bar(
                positions + offset,
                means,
                width=bar_width * 0.92,
                yerr=stds,
                capsize=1.3,
                error_kw={"elinewidth": 0.55, "ecolor": "#555555"},
                color=colors[series_index],
                edgecolor="white",
                linewidth=0.25,
                label=rf"$\eta$={_format_value(series_value)}",
            )
        axis.set_xlabel(PARAMETER_LABELS[x_key])
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions)
        axis.set_xticklabels([_format_value(value) for value in x_values])
        axis.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.3)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    axes[0].set_title(f"{DATASET_LABELS.get(dataset, dataset)} DP")
    axes[1].set_title(f"{DATASET_LABELS.get(dataset, dataset)} EO")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(4, len(series_values)),
        frameon=True,
        columnspacing=0.8,
        handlelength=1.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
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
    value_range = float(finite.max() - finite.min())
    padding = max(0.1, value_range * 0.12)
    bottom = max(0.0, float(finite.min()) - padding)
    upper = float(finite.max()) + padding
    normalizer = matplotlib.colors.Normalize(vmin=float(finite.min()), vmax=float(finite.max()) or 1.0)
    color_map = cm.get_cmap("RdYlBu_r")

    for y_index in range(len(y_values)):
        for x_index in range(len(x_values)):
            value = matrix[y_index, x_index]
            if not np.isfinite(value):
                continue
            axis.bar3d(
                x_index,
                y_index,
                bottom,
                0.62,
                0.62,
                max(0.0, float(value) - bottom),
                color=color_map(normalizer(float(value))),
                alpha=0.78,
                edgecolor="#777777",
                linewidth=0.18,
                shade=True,
            )

    axis.set_xticks(np.arange(len(x_values)) + 0.31)
    axis.set_xticklabels([_format_value(value) for value in x_values])
    axis.set_yticks(np.arange(len(y_values)) + 0.31)
    axis.set_yticklabels([_format_value(value) for value in y_values])
    # CELL's compact 3D panels use symbols instead of long axis sentences;
    # long labels are frequently clipped by Matplotlib's 3D bounding box.
    short_labels = {
        "lambda_coord": r"$\gamma$",
        "lambda_fair": r"$\beta$",
        "proto_temp": r"$\tau$",
        "lambda_pi": r"$\eta$",
    }
    axis.set_xlabel(short_labels.get(x_key, PARAMETER_LABELS[x_key]), labelpad=5)
    axis.set_ylabel(short_labels.get(y_key, PARAMETER_LABELS[y_key]), labelpad=5)
    axis.set_zlabel(z_label, labelpad=7)
    axis.set_zlim(bottom, upper)
    axis.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.tick_params(axis="x", pad=0, labelsize=8)
    axis.tick_params(axis="y", pad=0, labelsize=8)
    axis.tick_params(axis="z", pad=2, labelsize=8)
    axis.view_init(elev=25, azim=-52)


def plot_3d_fairness_bars(
    rows: Sequence[Mapping[str, object]],
    dataset: str,
    x_key: str,
    y_key: str,
    output_path: Path,
) -> None:
    """Draw paired 3D DP/EO bars like CELL's loss-weight analysis."""

    x_values = _sorted_values(rows, x_key)
    y_values = _sorted_values(rows, y_key)
    if not x_values or not y_values:
        return
    dp = _pair_matrix(
        rows, x_key, y_key, x_values, y_values, "target_after_dp_mean"
    )
    eo = _pair_matrix(
        rows, x_key, y_key, x_values, y_values, "target_after_eo_mean"
    )

    fig = plt.figure(figsize=(7.6, 3.35))
    dp_axis = fig.add_subplot(121, projection="3d")
    eo_axis = fig.add_subplot(122, projection="3d")
    _plot_3d_metric(dp_axis, dp, x_values, y_values, x_key, y_key, "DP (%)")
    _plot_3d_metric(eo_axis, eo, x_values, y_values, x_key, y_key, "EO (%)")
    dp_axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} DP", pad=0)
    eo_axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} EO", pad=0)
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.06, top=0.92, wspace=0.02)
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
                fairness_path = output_dir / f"tau_eta_{dataset}.pdf"
                heatmap_path = output_dir / f"tau_eta_{dataset}_composite.pdf"
                plot_grouped_fairness_bars(
                    group_rows,
                    dataset,
                    "proto_temp",
                    "lambda_pi",
                    fairness_path,
                )
                plot_composite_heatmap(
                    group_rows,
                    dataset,
                    group,
                    "proto_temp",
                    "lambda_pi",
                    heatmap_path,
                )
                generated.extend((fairness_path, heatmap_path))
            elif group == "beta_gamma":
                fairness_path = output_dir / f"beta_gamma_{dataset}.pdf"
                heatmap_path = output_dir / f"beta_gamma_{dataset}_composite.pdf"
                plot_3d_fairness_bars(
                    group_rows,
                    dataset,
                    "lambda_coord",
                    "lambda_fair",
                    fairness_path,
                )
                plot_composite_heatmap(
                    group_rows,
                    dataset,
                    group,
                    "lambda_coord",
                    "lambda_fair",
                    heatmap_path,
                )
                generated.extend((fairness_path, heatmap_path))

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
