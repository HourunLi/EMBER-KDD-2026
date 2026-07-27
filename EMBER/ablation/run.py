"""Run the five paper-aligned EMBER ablations and aggregate their metrics.

Each variant removes exactly one mechanism while retaining the rest of EMBER:
decoupled group recovery, meta-coordination, minority-aware update weighting,
Bayesian prior correction, or group-balanced prototype initialization.
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
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Edit these defaults, then run: python ablation/run.py
# ---------------------------------------------------------------------------
EXPERIMENT = "all"  # decouple | metaalign | minority | bca | groupinit | all
GPU_IDS = [0, 1, 2, 3, 5]
DATASETS = ["bailA"]
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
    """Execution and paper-mapping metadata for one single-module ablation."""

    name: str
    paper_variant: str
    variant: str
    paper_section: str
    equation: str
    challenge: str
    removed_module: str
    intervention: str


ABLATION_SPECS: Tuple[AblationSpec, ...] = (
    AblationSpec(
        name="decouple",
        paper_variant="var1",
        variant="wo_decouple",
        paper_section="3.2",
        equation="Eqs. (3)-(5), (11)-(12)",
        challenge="1(b)",
        removed_module="decoupled sensitive representation for group recovery",
        intervention=(
            "build and match group anchors in task space z instead of "
            "sensitive space e"
        ),
    ),
    AblationSpec(
        name="metaalign",
        paper_variant="var2",
        variant="wo_metaalign",
        paper_section="3.3",
        equation="Eqs. (7)-(9)",
        challenge="1(a)",
        removed_module="meta-coordinated fair alignment",
        intervention=(
            "set coordination strength gamma to zero while retaining "
            "class-conditional MMD"
        ),
    ),
    AblationSpec(
        name="minority",
        paper_variant="var3",
        variant="wo_minority",
        paper_section="3.4",
        equation="Eq. (14)",
        challenge="2(a)",
        removed_module="minority-aware inverse-frequency update weighting",
        intervention=(
            "use uniform group weights while retaining confidence weighting "
            "and residual evolution"
        ),
    ),
    AblationSpec(
        name="bca",
        paper_variant="var4",
        variant="wo_bca",
        paper_section="3.5",
        equation="Eqs. (17)-(18)",
        challenge="2(b)",
        removed_module="Bayesian target-prior correction",
        intervention=(
            "set prior strength eta to zero and predict from prototype "
            "likelihoods"
        ),
    ),
    AblationSpec(
        name="groupinit",
        paper_variant="var5",
        variant="wo_groupinit",
        paper_section="3.4",
        equation="Eq. (10)",
        challenge="2(a)",
        removed_module="group-balanced source prototype initialization",
        intervention=(
            "initialize each class prototype from its unbalanced source-node "
            "mean"
        ),
    ),
)
ABLATION_BY_NAME: Dict[str, AblationSpec] = {
    spec.name: spec for spec in ABLATION_SPECS
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
    ablation_names = tuple(ABLATION_BY_NAME)
    if EXPERIMENT not in {*ablation_names, "all"}:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")
    if EXPERIMENT == "all":
        return list(ABLATION_SPECS)
    return [ABLATION_BY_NAME[EXPERIMENT]]


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


def run_task(gpu_id: int, task: Tuple[str, int, int]) -> None:
    """Run the selected single-module ablations for one dataset/seed pair."""
    dataset, run_index, seed = task
    for spec in variant_specs():
        wait_for_gpu(gpu_id)
        run_variant(gpu_id, dataset, run_index, seed, spec)


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
                "paper_section": spec.paper_section,
                "equation": spec.equation,
                "challenge": spec.challenge,
                "removed_module": spec.removed_module,
                "intervention": spec.intervention,
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
    # Each dataset/seed task runs the same ordered five-ablation suite; GPU
    # workers provide scheduling and resource isolation only.
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
