#!/usr/bin/env python3
"""Run and plot the EMBER/SFFGNN parameter-analysis experiments.

The runner performs the three parameter groups recommended for the paper:

``delta``
    Confidence threshold ``tau_c`` (paper notation: delta).
``tau_eta``
    Prototype temperature ``proto_temp`` and Bayesian prior strength
    ``lambda_pi`` (paper notation: tau and eta).
``beta_gamma``
    Source fairness and coordination weights ``lambda_fair`` and
    ``lambda_coord`` (paper notation: beta and gamma).

Two supplementary one-dimensional groups are also available:
``lambda_residual_l2`` and ``adapt_epochs``.

Every subprocess writes one raw JSON result.  This script adds experiment
metadata, supports safe resume, aggregates repeated seeds into CSV/JSON, and
then calls ``pilot/parameter_analysis.py`` to generate polished PDF figures
under ``SFFGNN/parameter/results``.

Typical usage
-------------
Run the three paper groups on four datasets with four GPUs::

    python parameter/run.py --gpus 0 1 2 3

Run only the target-stage scans on GermanA and Pokec::

    python parameter/run.py --groups delta tau_eta \
        --datasets germanA pokec --gpus 0 1

Regenerate summaries and figures without launching experiments::

    python parameter/run.py --plot-only

Use ``--dry-run`` to inspect the complete experiment plan and commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


# ---------------------------------------------------------------------------
# Paper defaults.  They can be replaced from the command line without editing
# the file; keeping them here also makes a plain ``python parameter/run.py``
# reproducible.
# ---------------------------------------------------------------------------
DEFAULT_GROUPS = ("delta", "tau_eta", "beta_gamma")
DEFAULT_DATASETS = ("bailA", "germanA", "pokec", "syn")
DEFAULT_GPU_IDS = (0,)
DEFAULT_RUN_SEEDS = (1111, 2222, 3333, 4444, 5555)

DEFAULT_DELTA_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_TAU_VALUES = (0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_ETA_VALUES = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_BETA_VALUES = (0.0, 1.0, 2.0, 4.0, 8.0)
DEFAULT_GAMMA_VALUES = (0.0, 0.5, 1.0, 2.0, 3.0)
DEFAULT_RESIDUAL_L2_VALUES = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
DEFAULT_ADAPT_EPOCH_VALUES = (1, 10, 25, 50, 100, 150)

GPU_MAX_MEMORY_MB = 10000
GPU_MAX_UTILIZATION = 50
GPU_POLL_SECONDS = 30

GROUPS = (
    "delta",
    "tau_eta",
    "beta_gamma",
    "lambda_residual_l2",
    "adapt_epochs",
)

DATASET_DOMAINS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

PARAMETER_COLUMNS = (
    "tau_c",
    "proto_temp",
    "lambda_pi",
    "lambda_fair",
    "lambda_coord",
    "lambda_residual_l2",
    "adapt_epochs",
)

SCRIPT_DIR = Path(__file__).resolve().parent
SFFGNN_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SFFGNN_DIR.parent
MAIN_SCRIPT = SFFGNN_DIR / "main.py"
PLOT_SCRIPT = PROJECT_ROOT / "pilot" / "parameter_analysis.py"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExperimentSpec:
    """One point in a parameter-analysis grid."""

    group: str
    parameters: Tuple[Tuple[str, float | int], ...]

    @property
    def parameter_dict(self) -> Dict[str, float | int]:
        return dict(self.parameters)

    @property
    def experiment_id(self) -> str:
        parts = [self.group]
        for name, value in self.parameters:
            parts.append(f"{name}={slug_value(value)}")
        return "__".join(parts)


def print_status(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def slug_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{float(value):.8g}"
    return (
        text.replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def override_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def normalize_common_overrides(overrides: Sequence[str]) -> List[str]:
    normalized = []
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Invalid --override {item!r}; expected KEY=VALUE")
        normalized.append(f"{key.strip()}={value.strip()}")
    return normalized


def make_specs(args: argparse.Namespace) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    selected = list(dict.fromkeys(args.groups))
    if "delta" in selected:
        specs.extend(
            ExperimentSpec("delta", (("tau_c", value),))
            for value in args.delta_values
        )
    if "tau_eta" in selected:
        specs.extend(
            ExperimentSpec(
                "tau_eta",
                (("proto_temp", tau), ("lambda_pi", eta)),
            )
            for tau in args.tau_values
            for eta in args.eta_values
        )
    if "beta_gamma" in selected:
        specs.extend(
            ExperimentSpec(
                "beta_gamma",
                (("lambda_fair", beta), ("lambda_coord", gamma)),
            )
            for beta in args.beta_values
            for gamma in args.gamma_values
        )
    if "lambda_residual_l2" in selected:
        specs.extend(
            ExperimentSpec(
                "lambda_residual_l2",
                (("lambda_residual_l2", value),),
            )
            for value in args.residual_l2_values
        )
    if "adapt_epochs" in selected:
        specs.extend(
            ExperimentSpec("adapt_epochs", (("adapt_epochs", value),))
            for value in args.adapt_epoch_values
        )

    if args.max_specs is not None:
        specs = specs[: args.max_specs]
    return specs


def gpu_status(gpu_id: int) -> Tuple[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "nvidia-smi was not found; use --gpus -1 for CPU execution"
        ) from error
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        index, memory, utilization = parts[:3]
        if int(index) == int(gpu_id):
            return int(memory), int(utilization)
    raise RuntimeError(f"GPU {gpu_id} was not reported by nvidia-smi")


def wait_for_gpu(gpu_id: int, args: argparse.Namespace) -> None:
    if gpu_id < 0 or args.skip_gpu_wait:
        return
    while True:
        memory, utilization = gpu_status(gpu_id)
        if (
            memory <= args.gpu_max_memory_mb
            and utilization <= args.gpu_max_utilization
        ):
            return
        print_status(
            f"[GPU {gpu_id}] busy: memory={memory} MiB, "
            f"util={utilization}%; waiting"
        )
        time.sleep(args.gpu_poll_seconds)


def raw_result_path(
    results_dir: Path,
    dataset: str,
    spec: ExperimentSpec,
    seed: int,
) -> Path:
    return (
        results_dir
        / "raw"
        / spec.group
        / dataset
        / spec.experiment_id
        / f"seed_{seed}.json"
    )


def log_path(
    results_dir: Path,
    dataset: str,
    spec: ExperimentSpec,
    seed: int,
) -> Path:
    return (
        results_dir
        / "logs"
        / spec.group
        / dataset
        / spec.experiment_id
        / f"seed_{seed}.log"
    )


def expected_metadata(
    dataset: str,
    spec: ExperimentSpec,
    seed: int,
    target_seed: int,
    common_overrides: Sequence[str],
) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "group": spec.group,
        "experiment_id": spec.experiment_id,
        "parameters": spec.parameter_dict,
        "seed": seed,
        "target_seed": target_seed,
        "common_overrides": list(common_overrides),
    }


def valid_metric_payload(payload: Mapping[str, object]) -> bool:
    try:
        stages = payload["metrics"]
        for stage in ("source", "target_before", "target_after"):
            for metric in ("acc", "auc", "dp", "eo"):
                values = stages[stage][metric]
                if not values or not all(math.isfinite(float(value)) for value in values):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def build_command(
    args: argparse.Namespace,
    gpu_id: int,
    dataset: str,
    spec: ExperimentSpec,
    seed: int,
    target_seed: int,
    output_path: Path,
    output_log: Path,
    common_overrides: Sequence[str],
) -> List[str]:
    inid, outid = DATASET_DOMAINS[dataset]
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--dataset",
        dataset,
        "--inid",
        inid,
        "--outid",
        outid,
        "--device_id",
        str(gpu_id),
        "--seed",
        str(seed),
        "--target_seed",
        str(target_seed),
        "--runs_override",
        "1",
        "--ablation",
        "full",
        "--log_path",
        str(output_log),
        "--result_path",
        str(output_path),
        "--disable_embedding_export",
        "--use_checkpoint",
    ]
    for item in common_overrides:
        command.extend(("--override", item))
    for name, value in spec.parameters:
        command.extend(("--override", f"{name}={override_value(value)}"))
    return command


def annotate_result(
    path: Path,
    metadata: Mapping[str, object],
    command: Sequence[str],
) -> None:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not valid_metric_payload(payload):
        raise RuntimeError(f"Result file has incomplete metrics: {path}")
    payload["parameter_analysis"] = dict(metadata)
    payload["parameter_analysis"]["command"] = list(command)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)


def metadata_without_command(metadata: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in metadata.items() if key != "command"}


def result_matches(path: Path, metadata: Mapping[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        actual = payload.get("parameter_analysis", {})
        actual = metadata_without_command(actual)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return valid_metric_payload(payload) and actual == metadata


def run_experiment(
    args: argparse.Namespace,
    gpu_id: int,
    dataset: str,
    spec: ExperimentSpec,
    seed: int,
    run_index: int,
    common_overrides: Sequence[str],
) -> None:
    target_seed = seed + args.target_seed_offset
    output_path = raw_result_path(args.results_dir, dataset, spec, seed)
    output_log = log_path(args.results_dir, dataset, spec, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    metadata = expected_metadata(
        dataset,
        spec,
        seed,
        target_seed,
        common_overrides,
    )

    if not args.rerun and result_matches(output_path, metadata):
        print_status(
            f"[GPU {gpu_id}] skip completed {dataset} "
            f"seed={seed} {spec.experiment_id}"
        )
        return

    command = build_command(
        args,
        gpu_id,
        dataset,
        spec,
        seed,
        target_seed,
        output_path,
        output_log,
        common_overrides,
    )
    print_status(
        f"[GPU {gpu_id}] {dataset} run={run_index + 1} "
        f"seed={seed} {spec.experiment_id}"
    )
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=SFFGNN_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        stderr_path = output_log.with_suffix(".stderr.log")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        raise RuntimeError(
            f"Experiment failed: {dataset} seed={seed} {spec.experiment_id}\n"
            f"Log: {output_log}\nStderr: {stderr_path}"
        )
    if not output_path.exists():
        raise RuntimeError(f"Experiment did not create {output_path}")
    annotate_result(output_path, metadata, command)


def worker(
    gpu_id: int,
    tasks: queue.Queue,
    specs: Sequence[ExperimentSpec],
    args: argparse.Namespace,
    common_overrides: Sequence[str],
    failures: List[str],
    stop_event: threading.Event,
) -> None:
    while True:
        task = tasks.get()
        try:
            if task is None:
                return
            dataset, run_index, seed = task
            if stop_event.is_set() and args.fail_fast:
                continue
            wait_for_gpu(gpu_id, args)
            for spec in specs:
                if stop_event.is_set() and args.fail_fast:
                    break
                try:
                    run_experiment(
                        args,
                        gpu_id,
                        dataset,
                        spec,
                        seed,
                        run_index,
                        common_overrides,
                    )
                except Exception as error:  # keep other grid points resumable
                    message = str(error)
                    failures.append(message)
                    print_status(f"[GPU {gpu_id}] ERROR: {message}")
                    if args.fail_fast:
                        stop_event.set()
                        break
        finally:
            tasks.task_done()


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def metric_value(record: Mapping[str, object], stage: str, metric: str) -> float:
    values = record["metrics"][stage][metric]
    return statistics.fmean(float(value) for value in values)


def scan_raw_records(results_dir: Path) -> List[MutableMapping[str, object]]:
    raw_dir = results_dir / "raw"
    records: List[MutableMapping[str, object]] = []
    if not raw_dir.exists():
        return records
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as source:
                record = json.load(source)
        except (OSError, json.JSONDecodeError) as error:
            print_status(f"Ignoring unreadable result {path}: {error}")
            continue
        metadata = record.get("parameter_analysis")
        if not isinstance(metadata, dict) or not valid_metric_payload(record):
            print_status(f"Ignoring incomplete parameter result: {path}")
            continue
        record["_path"] = str(path)
        records.append(record)
    return records


def aggregate(results_dir: Path) -> Tuple[Path, Path, Path]:
    records = scan_raw_records(results_dir)
    run_rows: List[Dict[str, object]] = []
    for record in records:
        metadata = record["parameter_analysis"]
        parameters = metadata.get("parameters", {})
        row: Dict[str, object] = {
            "dataset": metadata["dataset"],
            "group": metadata["group"],
            "experiment_id": metadata["experiment_id"],
            "seed": metadata["seed"],
            "target_seed": metadata["target_seed"],
            "result_path": record["_path"],
        }
        for parameter in PARAMETER_COLUMNS:
            row[parameter] = parameters.get(parameter, "")
        for stage in ("source", "target_before", "target_after"):
            for metric in ("acc", "auc", "dp", "eo"):
                row[f"{stage}_{metric}"] = metric_value(record, stage, metric)
        row["composite"] = (
            row["target_after_acc"]
            + row["target_after_auc"]
            - row["target_after_dp"]
            - row["target_after_eo"]
        )
        run_rows.append(row)

    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in run_rows:
        key = (
            row["dataset"],
            row["group"],
            row["experiment_id"],
            *(row[column] for column in PARAMETER_COLUMNS),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        dataset, group, experiment_id, *parameter_values = key
        summary: Dict[str, object] = {
            "dataset": dataset,
            "group": group,
            "experiment_id": experiment_id,
            "runs_completed": len(group_rows),
        }
        summary.update(dict(zip(PARAMETER_COLUMNS, parameter_values)))
        for stage in ("source", "target_before", "target_after"):
            for metric in ("acc", "auc", "dp", "eo"):
                values = [float(row[f"{stage}_{metric}"]) for row in group_rows]
                mean, std = mean_std(values)
                summary[f"{stage}_{metric}_mean"] = mean
                summary[f"{stage}_{metric}_std"] = std
        composite_mean, composite_std = mean_std(
            [float(row["composite"]) for row in group_rows]
        )
        summary["composite_mean"] = composite_mean
        summary["composite_std"] = composite_std
        summary_rows.append(summary)

    results_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = results_dir / "runs.csv"
    summary_csv = results_dir / "summary.csv"
    summary_json = results_dir / "summary.json"

    run_fields = [
        "dataset",
        "group",
        "experiment_id",
        "seed",
        "target_seed",
        *PARAMETER_COLUMNS,
        *(
            f"{stage}_{metric}"
            for stage in ("source", "target_before", "target_after")
            for metric in ("acc", "auc", "dp", "eo")
        ),
        "composite",
        "result_path",
    ]
    summary_fields = [
        "dataset",
        "group",
        "experiment_id",
        "runs_completed",
        *PARAMETER_COLUMNS,
        *(
            f"{stage}_{metric}_{statistic}"
            for stage in ("source", "target_before", "target_after")
            for metric in ("acc", "auc", "dp", "eo")
            for statistic in ("mean", "std")
        ),
        "composite_mean",
        "composite_std",
    ]

    with runs_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with summary_json.open("w", encoding="utf-8") as output:
        json.dump(summary_rows, output, ensure_ascii=False, indent=2)

    print_status(
        f"Aggregated {len(run_rows)} runs into {len(summary_rows)} settings: "
        f"{summary_csv}"
    )
    return runs_csv, summary_csv, summary_json


def call_plotter(
    args: argparse.Namespace,
    summary_csv: Path,
) -> List[Path]:
    if not PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"Missing plotting script: {PLOT_SCRIPT}")
    command = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--summary",
        str(summary_csv),
        "--output-dir",
        str(args.results_dir),
        "--groups",
        *args.groups,
        "--datasets",
        *args.datasets,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Plotting failed with exit code {completed.returncode}\n"
            f"{completed.stderr}"
        )
    paths = [Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    for path in paths:
        print_status(f"Generated figure: {path}")
    if completed.stderr.strip():
        print_status(completed.stderr.strip())
    return paths


def write_manifest(
    args: argparse.Namespace,
    specs: Sequence[ExperimentSpec],
    common_overrides: Sequence[str],
) -> Path:
    path = args.results_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "groups": args.groups,
        "datasets": args.datasets,
        "gpus": args.gpus,
        "seeds": args.seeds,
        "common_overrides": list(common_overrides),
        "spec_count": len(specs),
        "total_subprocesses": len(specs) * len(args.datasets) * len(args.seeds),
        "specs": [
            {
                "group": spec.group,
                "experiment_id": spec.experiment_id,
                "parameters": spec.parameter_dict,
            }
            for spec in specs
        ],
    }
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    return path


def print_dry_run(
    args: argparse.Namespace,
    specs: Sequence[ExperimentSpec],
    common_overrides: Sequence[str],
) -> None:
    total = len(specs) * len(args.datasets) * len(args.seeds)
    print_status(
        f"Dry run: {len(specs)} settings x {len(args.datasets)} datasets x "
        f"{len(args.seeds)} seeds = {total} subprocesses"
    )
    for spec in specs:
        print_status(f"  {spec.experiment_id}: {spec.parameter_dict}")
    if specs and args.datasets and args.seeds:
        dataset = args.datasets[0]
        seed = args.seeds[0]
        spec = specs[0]
        path = raw_result_path(args.results_dir, dataset, spec, seed)
        log = log_path(args.results_dir, dataset, spec, seed)
        command = build_command(
            args,
            args.gpus[0],
            dataset,
            spec,
            seed,
            seed + args.target_seed_offset,
            path,
            log,
            common_overrides,
        )
        print_status("Example command:")
        print_status(subprocess.list2cmdline(command))


def validate_args(args: argparse.Namespace) -> None:
    if not args.groups:
        raise ValueError("At least one parameter group is required")
    if not args.datasets:
        raise ValueError("At least one dataset is required")
    unknown_datasets = set(args.datasets) - set(DATASET_DOMAINS)
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {sorted(unknown_datasets)}")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if not args.gpus:
        raise ValueError("At least one GPU id is required; use -1 for CPU")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus must not contain duplicates")
    if any(gpu_id < -1 for gpu_id in args.gpus):
        raise ValueError("GPU ids must be non-negative, or -1 for CPU")
    if -1 in args.gpus and len(args.gpus) != 1:
        raise ValueError("CPU id -1 cannot be combined with GPU ids")
    if any(value <= 0 for value in args.tau_values):
        raise ValueError("Prototype temperatures must be positive")
    if any(value < 0 for value in args.eta_values):
        raise ValueError("Prior strengths must be non-negative")
    if any(value < 0 or value > 1 for value in args.delta_values):
        raise ValueError("Confidence thresholds must lie in [0, 1]")
    if any(value < 1 for value in args.adapt_epoch_values):
        raise ValueError(
            "adapt_epochs must be at least 1; use target_before metrics as the T=0 reference"
        )
    if args.max_specs is not None and args.max_specs < 1:
        raise ValueError("--max-specs must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SFFGNN parameter analysis and generate publication-ready PDFs."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=GROUPS,
        default=list(DEFAULT_GROUPS),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_DOMAINS),
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=list(DEFAULT_GPU_IDS),
        help="GPU ids; use -1 to run a single CPU worker",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_RUN_SEEDS),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--delta-values", nargs="+", type=float, default=list(DEFAULT_DELTA_VALUES))
    parser.add_argument("--tau-values", nargs="+", type=float, default=list(DEFAULT_TAU_VALUES))
    parser.add_argument("--eta-values", nargs="+", type=float, default=list(DEFAULT_ETA_VALUES))
    parser.add_argument("--beta-values", nargs="+", type=float, default=list(DEFAULT_BETA_VALUES))
    parser.add_argument("--gamma-values", nargs="+", type=float, default=list(DEFAULT_GAMMA_VALUES))
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
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="common SFFGNN override applied before the varied parameters",
    )
    parser.add_argument("--target-seed-offset", type=int, default=100000)
    parser.add_argument("--gpu-max-memory-mb", type=int, default=GPU_MAX_MEMORY_MB)
    parser.add_argument("--gpu-max-utilization", type=int, default=GPU_MAX_UTILIZATION)
    parser.add_argument("--gpu-poll-seconds", type=int, default=GPU_POLL_SECONDS)
    parser.add_argument("--skip-gpu-wait", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="aggregate existing raw results and generate figures",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="aggregate existing raw results without plotting",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--max-specs",
        type=int,
        help="truncate the parameter grid for orchestration smoke tests",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.groups = list(dict.fromkeys(args.groups))
    args.datasets = list(dict.fromkeys(args.datasets))
    args.results_dir = args.results_dir.resolve()
    validate_args(args)
    common_overrides = normalize_common_overrides(args.override)
    specs = make_specs(args)
    if not specs and not (args.plot_only or args.aggregate_only):
        raise ValueError("The selected parameter grids contain no settings")

    if args.dry_run:
        print_dry_run(args, specs, common_overrides)
        return

    failures: List[str] = []
    if not args.plot_only and not args.aggregate_only:
        write_manifest(args, specs, common_overrides)
        if not MAIN_SCRIPT.exists():
            raise FileNotFoundError(f"Missing SFFGNN entry point: {MAIN_SCRIPT}")
        if not args.skip_gpu_wait:
            for gpu_id in args.gpus:
                if gpu_id >= 0:
                    gpu_status(gpu_id)

        tasks: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        for dataset in args.datasets:
            for run_index, seed in enumerate(args.seeds):
                tasks.put((dataset, run_index, seed))
        for _ in args.gpus:
            tasks.put(None)

        workers = [
            threading.Thread(
                target=worker,
                args=(
                    gpu_id,
                    tasks,
                    specs,
                    args,
                    common_overrides,
                    failures,
                    stop_event,
                ),
                daemon=True,
            )
            for gpu_id in args.gpus
        ]
        for thread in workers:
            thread.start()
        tasks.join()
        for thread in workers:
            thread.join()
        if failures:
            print_status(
                f"Completed with {len(failures)} failed experiment(s); "
                f"successful settings remain resumable."
            )

    _, summary_csv, _ = aggregate(args.results_dir)
    if args.aggregate_only or args.no_plot:
        if failures:
            raise RuntimeError(
                f"{len(failures)} parameter experiment(s) failed; "
                f"see logs under {args.results_dir / 'logs'}"
            )
        return
    if not scan_raw_records(args.results_dir):
        if failures:
            raise RuntimeError(
                f"All parameter experiments failed; see {args.results_dir / 'logs'}"
            )
        raise RuntimeError(
            f"No completed parameter results were found under {args.results_dir / 'raw'}"
        )
    call_plotter(args, summary_csv)
    if failures:
        raise RuntimeError(
            f"{len(failures)} parameter experiment(s) failed; "
            f"partial summaries and figures were retained"
        )


if __name__ == "__main__":
    main()
