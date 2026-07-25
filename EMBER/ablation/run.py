"""Run and aggregate the four EMBER ablation variants.

Each ablation has an explicit runner so its meaning is visible at the call
site. ``run_all_ablations`` is the single batch entry point used for every
dataset/seed task. Variant 3 removes the historical part of the prior update
in Eq. (17) (equivalently, omega=0); it does not disable the cumulative
prototype average in Eq. (16).
"""

from __future__ import annotations

import csv
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Edit these defaults, then run: python ablation/run.py
# ---------------------------------------------------------------------------
EXPERIMENT = "all"  # metaalign | bca | ema | residual | all
GPU_IDS = [0, 2, 4, 6, 7]
DATASETS = ["pokec"]
RUN_SEEDS = [1111, 2222, 3333, 4444, 5555]

GPU_MAX_MEMORY_MB = 10000
GPU_MAX_UTILIZATION = 50
GPU_POLL_SECONDS = 30

DATASET_DOMAINS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}


@dataclass(frozen=True)
class AblationSpec:
    """Stable identifiers and reporting metadata for one ablation."""

    name: str
    paper_variant: str
    variant: str
    removed_module: str


METAALIGN_SPEC = AblationSpec(
    name="metaalign",
    paper_variant="var1",
    variant="wo_metaalign",
    removed_module="meta-coordinated fair alignment",
)
BCA_SPEC = AblationSpec(
    name="bca",
    paper_variant="var2",
    variant="wo_bca",
    removed_module="Bayesian prior correction",
)
EMA_SPEC = AblationSpec(
    name="ema",
    paper_variant="var3",
    variant="wo_ema",
    removed_module=(
        "discounted prior-evidence history in Eq. (17) "
        "(omega=0; current-round prior retained)"
    ),
)
RESIDUAL_SPEC = AblationSpec(
    name="residual",
    paper_variant="var4",
    variant="wo_residual",
    removed_module="minority-aware residual evolution",
)

ABLATION_SPECS: Dict[str, AblationSpec] = {
    spec.name: spec
    for spec in (METAALIGN_SPEC, BCA_SPEC, EMA_SPEC, RESIDUAL_SPEC)
}


SCRIPT_DIR = Path(__file__).resolve().parent
EMBER_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
LOG_DIR = RESULTS_DIR / "logs"
PRINT_LOCK = threading.Lock()


def print_status(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def variant_specs() -> List[AblationSpec]:
    """Return the variants selected by ``EXPERIMENT`` in paper order."""
    ablation_names = tuple(ABLATION_SPECS)
    if EXPERIMENT not in {*ablation_names, "all"}:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")
    if EXPERIMENT == "all":
        return [ABLATION_SPECS[name] for name in ablation_names]
    return [ABLATION_SPECS[EXPERIMENT]]


def gpu_status(gpu_id: int) -> Tuple[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    for line in completed.stdout.strip().splitlines():
        index, memory, utilization = [part.strip() for part in line.split(",")[:3]]
        if int(index) == int(gpu_id):
            return int(memory), int(utilization)
    raise RuntimeError(f"GPU {gpu_id} was not reported by nvidia-smi")


def wait_for_gpu(gpu_id: int) -> None:
    while True:
        memory, utilization = gpu_status(gpu_id)
        if memory <= GPU_MAX_MEMORY_MB and utilization <= GPU_MAX_UTILIZATION:
            return
        print_status(
            f"[GPU {gpu_id}] busy: memory={memory} MiB, util={utilization}%; waiting"
        )
        time.sleep(GPU_POLL_SECONDS)


def result_path(dataset: str, variant: str, run_index: int) -> Path:
    return RAW_DIR / dataset / variant / f"run_{run_index + 1}.json"


def run_variant(
    gpu_id: int,
    dataset: str,
    run_index: int,
    seed: int,
    spec: AblationSpec,
) -> None:
    """Execute one seed of one dataset for the supplied ablation spec."""
    inid, outid = DATASET_DOMAINS[dataset]
    output_path = result_path(dataset, spec.variant, run_index)
    log_path = LOG_DIR / dataset / spec.variant / f"run_{run_index + 1}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(EMBER_DIR / "main.py"),
        "--dataset", dataset,
        "--inid", inid,
        "--outid", outid,
        "--device_id", str(gpu_id),
        "--seed", str(seed),
        "--target_seed", str(seed + 100000),
        "--runs_override", "1",
        "--ablation", spec.name,
        "--log_path", str(log_path),
        "--result_path", str(output_path),
        "--disable_embedding_export",
    ]
    print_status(
        f"[GPU {gpu_id}] {dataset} run={run_index + 1} "
        f"variant={spec.variant} seed={seed}"
    )
    completed = subprocess.run(
        command,
        cwd=EMBER_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{dataset} run {run_index + 1} {spec.variant} failed\n"
            f"Log: {log_path}\n{completed.stderr}"
        )
    if not output_path.exists():
        raise RuntimeError(f"Missing result file: {output_path}")


def run_metaalign_ablation(
    gpu_id: int,
    dataset: str,
    run_index: int,
    seed: int,
) -> None:
    """Variant 1: remove source meta-coordinated fair alignment."""
    run_variant(gpu_id, dataset, run_index, seed, METAALIGN_SPEC)


def run_bca_ablation(
    gpu_id: int,
    dataset: str,
    run_index: int,
    seed: int,
) -> None:
    """Variant 2: remove the Bayesian prior correction."""
    run_variant(gpu_id, dataset, run_index, seed, BCA_SPEC)


def run_ema_ablation(
    gpu_id: int,
    dataset: str,
    run_index: int,
    seed: int,
) -> None:
    """Variant 3: set omega=0 in Eq. (17), removing prior history only."""
    run_variant(gpu_id, dataset, run_index, seed, EMA_SPEC)


def run_residual_ablation(
    gpu_id: int,
    dataset: str,
    run_index: int,
    seed: int,
) -> None:
    """Variant 4: remove minority-aware residual prototype evolution."""
    run_variant(gpu_id, dataset, run_index, seed, RESIDUAL_SPEC)


ABLATION_RUNNERS: Dict[str, Callable[[int, str, int, int], None]] = {
    "metaalign": run_metaalign_ablation,
    "bca": run_bca_ablation,
    "ema": run_ema_ablation,
    "residual": run_residual_ablation,
}


def run_all_ablations(
    gpu_id: int,
    dataset: str,
    run_index: int,
    seed: int,
) -> None:
    """Batch entry point: invoke all four ablation methods for one task."""
    run_metaalign_ablation(gpu_id, dataset, run_index, seed)
    run_bca_ablation(gpu_id, dataset, run_index, seed)
    run_ema_ablation(gpu_id, dataset, run_index, seed)
    run_residual_ablation(gpu_id, dataset, run_index, seed)


def run_task(gpu_id: int, task: Tuple[str, int, int]) -> None:
    dataset, run_index, seed = task
    wait_for_gpu(gpu_id)
    if EXPERIMENT == "all":
        run_all_ablations(gpu_id, dataset, run_index, seed)
    else:
        ABLATION_RUNNERS[EXPERIMENT](gpu_id, dataset, run_index, seed)


def worker(
    gpu_id: int,
    tasks: queue.Queue,
    failures: List[str],
) -> None:
    while True:
        task = tasks.get()
        try:
            if task is None:
                return
            run_task(gpu_id, task)
        except Exception as error:
            failures.append(str(error))
            print_status(f"[GPU {gpu_id}] ERROR: {error}")
        finally:
            tasks.task_done()


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    return statistics.fmean(values), statistics.pstdev(values)


def aggregate(specs: Sequence[AblationSpec]) -> None:
    rows = []
    stages = ("source", "target_before", "target_after")
    metrics = ("acc", "auc", "dp", "eo")
    for dataset in DATASETS:
        for spec in specs:
            records = []
            for run_index in range(len(RUN_SEEDS)):
                path = result_path(dataset, spec.variant, run_index)
                if path.exists():
                    with path.open("r", encoding="utf-8") as result_file:
                        records.append(json.load(result_file))
            if not records:
                continue

            row = {
                "dataset": dataset,
                "variant": spec.variant,
                "ablation": spec.name,
                "paper_variant": spec.paper_variant,
                "removed_module": spec.removed_module,
                "runs": len(records),
            }
            for stage in stages:
                for metric in metrics:
                    values = [
                        record["metrics"][stage][metric][0]
                        for record in records
                    ]
                    mean, std = mean_std(values)
                    row[f"{stage}_{metric}_mean"] = mean
                    row[f"{stage}_{metric}_std"] = std
            rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "summary.json"
    csv_path = RESULTS_DIR / "summary.csv"
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(rows, output, ensure_ascii=False, indent=2)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print_status(f"Saved summaries to {csv_path} and {json_path}")


def main() -> None:
    if not GPU_IDS:
        raise ValueError("GPU_IDS must contain at least one GPU")
    if len(set(GPU_IDS)) != len(GPU_IDS):
        raise ValueError("GPU_IDS must not contain duplicates")
    unknown_datasets = set(DATASETS) - set(DATASET_DOMAINS)
    if unknown_datasets:
        raise ValueError(f"Missing domain mapping for: {sorted(unknown_datasets)}")
    if len(RUN_SEEDS) != 5:
        raise ValueError("RUN_SEEDS must contain exactly five seeds")
    specs = variant_specs()
    for gpu_id in GPU_IDS:
        gpu_status(gpu_id)

    tasks = queue.Queue()
    failures = []
    # Four dataset tasks call the same four-method batch entry point for every
    # seed, while GPU workers only provide scheduling and resource isolation.
    for dataset in DATASETS:
        for run_index, seed in enumerate(RUN_SEEDS):
            tasks.put((dataset, run_index, seed))
    for _ in GPU_IDS:
        tasks.put(None)

    workers = [
        threading.Thread(
            target=worker,
            args=(gpu_id, tasks, failures),
            daemon=True,
        )
        for gpu_id in GPU_IDS
    ]
    for thread in workers:
        thread.start()
    tasks.join()
    for thread in workers:
        thread.join()

    aggregate(specs)
    if failures:
        raise RuntimeError(
            f"{len(failures)} run task(s) failed; see logs under {LOG_DIR}"
        )


if __name__ == "__main__":
    main()
