"""Run FairMAC ablations on idle GPUs and aggregate five repeated runs."""

import csv
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Edit these defaults, then run: python ablation/run_ablation.py
# ---------------------------------------------------------------------------
EXPERIMENT = "all"  # full | metaalign | bca | target_mmd | residual | all
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
DATASETS = ["bailA", "germanA", "pokec", "syn"]
RUN_SEEDS = [1111, 2222, 3333, 4444, 5555]

GPU_MAX_MEMORY_MB = 1000
GPU_MAX_UTILIZATION = 10
GPU_POLL_SECONDS = 30

DATASET_DOMAINS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}


SCRIPT_DIR = Path(__file__).resolve().parent
SFFGNN_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
LOG_DIR = RESULTS_DIR / "logs"
PRINT_LOCK = threading.Lock()


def print_status(message):
    with PRINT_LOCK:
        print(message, flush=True)


def variant_specs():
    if EXPERIMENT not in {"full", "metaalign", "bca", "target_mmd", "residual", "all"}:
        raise ValueError(f"Unknown EXPERIMENT: {EXPERIMENT}")

    selected = (
        ["full", "metaalign", "bca", "target_mmd", "residual"]
        if EXPERIMENT == "all"
        else [EXPERIMENT]
    )
    return [
        {
            "variant": "wo_" + name if name != "full" else "full",
            "ablation": name,
        }
        for name in selected
    ]


def gpu_status(gpu_id):
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


def wait_for_gpu(gpu_id):
    while True:
        memory, utilization = gpu_status(gpu_id)
        if memory <= GPU_MAX_MEMORY_MB and utilization <= GPU_MAX_UTILIZATION:
            return
        print_status(
            f"[GPU {gpu_id}] busy: memory={memory} MiB, util={utilization}%; waiting"
        )
        time.sleep(GPU_POLL_SECONDS)


def result_path(dataset, variant, run_index):
    return RAW_DIR / dataset / variant / f"run_{run_index + 1}.json"


def run_variant(gpu_id, dataset, run_index, seed, spec):
    inid, outid = DATASET_DOMAINS[dataset]
    output_path = result_path(dataset, spec["variant"], run_index)
    log_path = LOG_DIR / dataset / spec["variant"] / f"run_{run_index + 1}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(SFFGNN_DIR / "main.py"),
        "--dataset", dataset,
        "--inid", inid,
        "--outid", outid,
        "--device_id", str(gpu_id),
        "--seed", str(seed),
        "--target_seed", str(seed + 100000),
        "--runs_override", "1",
        "--ablation", spec["ablation"],
        "--log_path", str(log_path),
        "--result_path", str(output_path),
        "--disable_embedding_export",
    ]
    print_status(
        f"[GPU {gpu_id}] {dataset} run={run_index + 1} "
        f"variant={spec['variant']} seed={seed}"
    )
    completed = subprocess.run(
        command,
        cwd=SFFGNN_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{dataset} run {run_index + 1} {spec['variant']} failed\n"
            f"Log: {log_path}\n{completed.stderr}"
        )
    if not output_path.exists():
        raise RuntimeError(f"Missing result file: {output_path}")


def run_task(gpu_id, task, specs):
    dataset, run_index, seed = task
    wait_for_gpu(gpu_id)
    for spec in specs:
        run_variant(gpu_id, dataset, run_index, seed, spec)


def worker(gpu_id, tasks, specs, failures):
    while True:
        task = tasks.get()
        try:
            if task is None:
                return
            run_task(gpu_id, task, specs)
        except Exception as error:
            failures.append(str(error))
            print_status(f"[GPU {gpu_id}] ERROR: {error}")
        finally:
            tasks.task_done()


def mean_std(values):
    return statistics.fmean(values), statistics.pstdev(values)


def aggregate(specs):
    rows = []
    stages = ("source", "target_before", "target_after")
    metrics = ("acc", "auc", "dp", "eo")
    for dataset in DATASETS:
        for spec in specs:
            records = []
            for run_index in range(len(RUN_SEEDS)):
                path = result_path(dataset, spec["variant"], run_index)
                if path.exists():
                    with path.open("r", encoding="utf-8") as result_file:
                        records.append(json.load(result_file))
            if not records:
                continue

            row = {
                "dataset": dataset,
                "variant": spec["variant"],
                "ablation": spec["ablation"],
                "runs": len(records),
            }
            for stage in stages:
                for metric in metrics:
                    values = [record["metrics"][stage][metric][0] for record in records]
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


def main():
    if not GPU_IDS:
        raise ValueError("GPU_IDS must contain at least one GPU")
    if len(set(GPU_IDS)) != len(GPU_IDS):
        raise ValueError("GPU_IDS must not contain duplicates")
    unknown_datasets = set(DATASETS) - set(DATASET_DOMAINS)
    if unknown_datasets:
        raise ValueError(f"Missing domain mapping for: {sorted(unknown_datasets)}")
    if len(RUN_SEEDS) != 5:
        raise ValueError("RUN_SEEDS must contain exactly five seeds")
    for gpu_id in GPU_IDS:
        gpu_status(gpu_id)

    specs = variant_specs()
    tasks = queue.Queue()
    failures = []
    for dataset in DATASETS:
        for run_index, seed in enumerate(RUN_SEEDS):
            tasks.put((dataset, run_index, seed))
    for _ in GPU_IDS:
        tasks.put(None)

    workers = [
        threading.Thread(
            target=worker,
            args=(gpu_id, tasks, specs, failures),
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
        raise RuntimeError(f"{len(failures)} run task(s) failed; see logs under {LOG_DIR}")


if __name__ == "__main__":
    main()
