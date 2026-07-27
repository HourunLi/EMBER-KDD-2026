#!/usr/bin/env python3
"""Filter parameter summaries and render selected EMBER analysis figures.

``run.py`` deliberately aggregates every valid JSON below ``results/raw``.  That
is convenient when extending a grid, but it also means historical grid points
remain in the aggregate and can reappear in figures.  This script creates a
group-specific, value-specific summary before invoking
``pilot/parameter_analysis.py``.  It never changes raw experiment results.

Examples
--------
Render the default three paper groups with the same grids as ``run.py``::

    python EMBER/parameter/draw.py --method EMBER

Render a BailA-only refined nu0-delta grid::

    python EMBER/parameter/draw.py --method EMBER --datasets bailA \
        --groups nu0_delta --nu0-values 0.1 1 10 100 800 1000 \
        --delta-values 0.5 0.6 0.7 0.75 0.8 0.9

The resulting PDFs are written below ``EMBER/parameter/pdf/{method}/{group}``.
The exact filtered CSV supplied to the plotter is retained under
``EMBER/parameter/pdf/{method}/_filtered_summaries`` for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


# Keep these paper groups and their default grids in lockstep with run.py.
PAPER_GROUPS = ("beta_gamma", "nu0_delta", "eta_omega")
SUPPLEMENTARY_GROUPS = ("lambda_residual_l2", "adapt_epochs")
GROUPS = PAPER_GROUPS + SUPPLEMENTARY_GROUPS

DEFAULT_GROUPS = PAPER_GROUPS
DEFAULT_DATASETS = ("bailA", "pokec", "syn")

DEFAULT_NU0_VALUES = (0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_DELTA_VALUES = (0.5, 0.6, 0.7, 0.8, 0.9)

DEFAULT_ETA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_OMEGA_VALUES = (0.0, 0.3, 0.6, 0.8, 0.9, 0.99)

DEFAULT_BETA_VALUES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_GAMMA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)

DEFAULT_RESIDUAL_L2_VALUES = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
DEFAULT_ADAPT_EPOCH_VALUES = (1, 10, 25, 50, 100, 150)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PLOT_SCRIPT = PROJECT_ROOT / "pilot" / "parameter_analysis.py"
DEFAULT_SUMMARY = SCRIPT_DIR / "results" / "summary.csv"
DEFAULT_PDF_ROOT = SCRIPT_DIR / "pdf"


@dataclass(frozen=True)
class FilterSpec:
    """One exact parameter setting retained in a filtered summary."""

    group: str
    parameters: Tuple[Tuple[str, float], ...]

    @property
    def parameter_dict(self) -> Dict[str, float]:
        return dict(self.parameters)


def _float_key(value: object) -> float:
    """Parse one CSV numeric field with a clear error for an invalid row."""

    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a numeric value, got {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Expected a finite numeric value, got {value!r}")
    return parsed


def _same_value(left: object, right: float) -> bool:
    """Compare CSV floats without relying on their decimal string formatting."""

    try:
        return math.isclose(_float_key(left), right, rel_tol=0.0, abs_tol=1e-10)
    except ValueError:
        return False


def _validate_positive(values: Iterable[float], label: str) -> None:
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{label} must contain only positive finite values")


def _validate_unit_interval(
    values: Iterable[float], label: str, *, include_one: bool = True
) -> None:
    upper_valid = (lambda value: value <= 1.0) if include_one else (lambda value: value < 1.0)
    if any(
        not math.isfinite(value) or value < 0.0 or not upper_valid(value)
        for value in values
    ):
        suffix = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{label} must lie in {suffix}")


def make_specs(args: argparse.Namespace) -> List[FilterSpec]:
    """Build the exact Cartesian settings that should survive filtering."""

    selected = list(dict.fromkeys(args.groups))
    specs: List[FilterSpec] = []
    if "beta_gamma" in selected:
        specs.extend(
            FilterSpec("beta_gamma", (("lambda_fair", beta), ("lambda_coord", gamma)))
            for beta in args.beta_values
            for gamma in args.gamma_values
        )
    if "nu0_delta" in selected:
        specs.extend(
            FilterSpec(
                "nu0_delta", (("group_pseudocount", nu0), ("tau_c", delta))
            )
            for nu0 in args.nu0_values
            for delta in args.delta_values
        )
    if "eta_omega" in selected:
        specs.extend(
            FilterSpec(
                "eta_omega", (("lambda_pi", eta), ("prior_discount", omega))
            )
            for eta in args.eta_values
            for omega in args.omega_values
        )
    if "lambda_residual_l2" in selected:
        specs.extend(
            FilterSpec("lambda_residual_l2", (("lambda_residual_l2", value),))
            for value in args.residual_l2_values
        )
    if "adapt_epochs" in selected:
        specs.extend(
            FilterSpec("adapt_epochs", (("adapt_epochs", float(value)),))
            for value in args.adapt_epoch_values
        )
    return specs


def row_matches_spec(row: Mapping[str, str], spec: FilterSpec) -> bool:
    return row.get("group") == spec.group and all(
        _same_value(row.get(name, ""), value) for name, value in spec.parameters
    )


def load_summary(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Summary CSV not found: {path}. Run parameter/run.py first, or pass --summary."
        )
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"Summary CSV has no header: {path}")
        required = {"dataset", "group"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Summary CSV is missing required columns: {sorted(missing)}")
        return list(reader.fieldnames), list(reader)


def filter_rows(
    rows: Sequence[Mapping[str, str]],
    datasets: Sequence[str],
    specs: Sequence[FilterSpec],
) -> Dict[str, List[Dict[str, str]]]:
    """Return selected rows grouped by parameter group."""

    wanted_datasets = set(datasets)
    specs_by_group: Dict[str, List[FilterSpec]] = {}
    for spec in specs:
        specs_by_group.setdefault(spec.group, []).append(spec)

    selected: Dict[str, List[Dict[str, str]]] = {
        group: [] for group in specs_by_group
    }
    for row in rows:
        group = row.get("group", "")
        if row.get("dataset") not in wanted_datasets or group not in specs_by_group:
            continue
        if any(row_matches_spec(row, spec) for spec in specs_by_group[group]):
            selected[group].append(dict(row))
    return selected


def expected_settings(specs: Sequence[FilterSpec], group: str) -> int:
    return sum(spec.group == group for spec in specs)


def discovered_settings(rows: Sequence[Mapping[str, str]], group: str) -> int:
    settings = set()
    for row in rows:
        if row.get("group") != group:
            continue
        parameter_values = tuple(
            sorted(
                (name, str(row.get(name, "")))
                for name in (
                    "lambda_fair",
                    "lambda_coord",
                    "group_pseudocount",
                    "tau_c",
                    "lambda_pi",
                    "prior_discount",
                    "lambda_residual_l2",
                    "adapt_epochs",
                )
                if str(row.get(name, "")).strip()
            )
        )
        settings.add(parameter_values)
    return len(settings)


def write_filtered_summary(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_plotter(
    summary_path: Path,
    output_dir: Path,
    group: str,
    datasets: Sequence[str],
) -> List[Path]:
    command = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--summary",
        str(summary_path),
        "--output-dir",
        str(output_dir),
        "--groups",
        group,
        "--datasets",
        *datasets,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Plotting {group} failed with exit code {completed.returncode}:\n"
            f"{completed.stderr.strip()}"
        )
    generated = [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return generated


def validate_args(args: argparse.Namespace) -> None:
    if not args.groups:
        raise ValueError("At least one parameter group is required")
    if not args.datasets:
        raise ValueError("At least one dataset is required")
    if not args.method.strip() or any(token in args.method for token in ("/", "\\", ".")):
        raise ValueError("--method must be a simple directory name without '.', '/' or '\\'")
    _validate_unit_interval(args.delta_values, "delta values")
    _validate_unit_interval(args.eta_values, "eta values")
    _validate_unit_interval(args.gamma_values, "gamma values")
    _validate_unit_interval(args.omega_values, "omega values", include_one=False)
    _validate_positive(args.nu0_values, "nu0 values")
    if any(not math.isfinite(value) or value < 0.0 for value in args.beta_values):
        raise ValueError("beta values must be finite and non-negative")
    if any(not math.isfinite(value) or value < 0.0 for value in args.residual_l2_values):
        raise ValueError("residual L2 values must be finite and non-negative")
    if any(value < 1 for value in args.adapt_epoch_values):
        raise ValueError("adaptation epoch values must be at least 1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter an EMBER parameter summary and render selected PDF figures."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--method", default="EMBER")
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(DEFAULT_GROUPS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--beta-values", nargs="+", type=float, default=list(DEFAULT_BETA_VALUES))
    parser.add_argument("--gamma-values", nargs="+", type=float, default=list(DEFAULT_GAMMA_VALUES))
    parser.add_argument("--nu0-values", nargs="+", type=float, default=list(DEFAULT_NU0_VALUES))
    parser.add_argument("--delta-values", nargs="+", type=float, default=list(DEFAULT_DELTA_VALUES))
    parser.add_argument("--eta-values", nargs="+", type=float, default=list(DEFAULT_ETA_VALUES))
    parser.add_argument("--omega-values", nargs="+", type=float, default=list(DEFAULT_OMEGA_VALUES))
    parser.add_argument(
        "--residual-l2-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_RESIDUAL_L2_VALUES),
    )
    parser.add_argument(
        "--adapt-epoch-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_ADAPT_EPOCH_VALUES),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail instead of drawing when a selected dataset/group has missing settings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report selected row coverage without writing CSVs or PDFs",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.groups = list(dict.fromkeys(args.groups))
    args.datasets = list(dict.fromkeys(args.datasets))
    args.summary = args.summary.resolve()
    args.pdf_root = args.pdf_root.resolve()
    validate_args(args)

    fieldnames, rows = load_summary(args.summary)
    specs = make_specs(args)
    selected_by_group = filter_rows(rows, args.datasets, specs)

    for group in args.groups:
        group_rows = selected_by_group.get(group, [])
        required = expected_settings(specs, group)
        for dataset in args.datasets:
            dataset_rows = [row for row in group_rows if row.get("dataset") == dataset]
            found = discovered_settings(dataset_rows, group)
            message = f"{group} / {dataset}: {found}/{required} selected settings present"
            if found < required:
                message += " (some requested settings have no summary row)"
            print(message)
            if args.require_complete and found < required:
                raise RuntimeError(message)

        if not group_rows:
            raise RuntimeError(
                f"No summary rows match group {group!r}, datasets {args.datasets}, "
                "and the requested parameter grid."
            )
        if args.dry_run:
            continue

        method_root = args.pdf_root / args.method
        filtered_path = method_root / "_filtered_summaries" / f"{group}.csv"
        output_dir = method_root / group
        write_filtered_summary(filtered_path, fieldnames, group_rows)
        generated = run_plotter(filtered_path, output_dir, group, args.datasets)
        print(f"Filtered summary: {filtered_path}")
        for path in generated:
            print(f"Generated figure: {path}")


if __name__ == "__main__":
    main()
