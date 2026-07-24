"""Stage 1: tune only EMBER's source model on source validation data.

Every source configuration is trained twice with independent source seeds.  The
target graph is not loaded.  Results are ranked by

    ACC + AUC - DP - EO

on the source validation split.  After this pass, keep the top few source
packages and run a separate target-adaptation search with each package frozen.

Examples
--------
python tune.py --dry-run --datasets germanA --trials 3 --gpus 0
python tune.py --gpus 0 1 2 3 4 5
python tune.py --datasets pokec --trials 24 --gpus 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.py"
CONFIG = ROOT / "config" / "config.yaml"
DEFAULT_RESULTS_DIR = ROOT / "tune_results_source"
SEARCH_VERSION = "source_stage_v1"
DEFAULT_SEEDS = (1111, 2222)

DATASETS = {
    "bailA": {
        "domains": ("_2", "_1"),
        "trials": 20,
        "min_meta_step": 0.009,
        "min_free_mb": 8000,
    },
    "germanA": {
        "domains": ("_2", "_1"),
        "trials": 20,
        "min_meta_step": 0.009,
        "min_free_mb": 6000,
    },
    "pokec": {
        "domains": ("_z", "_n"),
        "trials": 20,
        "min_meta_step": 0.012,
        "min_free_mb": 18000,
    },
    "syn": {
        "domains": ("-2", "-1"),
        "trials": 20,
        "min_meta_step": 0.006,
        "min_free_mb": 8000,
    },
}

# Numerical/runtime settings are fixed.  They are not method-selection knobs.
FIXED = {
    "train_epochs": 800,
    "disentangle_batch_size": 512,
    "source_mmd_min_samples": 2,
    "source_mmd_max_samples": 1024,
    "mmd_chunk_size": 256,
}

SOURCE_ORDER = (
    "inter_encoder",
    "hidden_dim",
    "n_layers",
    "lr",
    "lr2_reg",
    "dropout",
    "train_epochs",
    "lambda_fair",
    "meta_lr",
    "lambda_coord",
    "lambda_sen",
    "lambda_dec",
    "disentangle_batch_size",
    "source_mmd_bandwidth",
    "source_mmd_min_samples",
    "source_mmd_max_samples",
    "mmd_chunk_size",
)


def values(key: str, items: Iterable[Any]) -> tuple[dict[str, Any], ...]:
    return tuple({key: item} for item in items)


def architectures(*rows: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    choices = []
    for row in rows:
        encoder, hidden, *layers = row
        choice = {"inter_encoder": encoder, "hidden_dim": hidden}
        if layers:
            choice["n_layers"] = layers[0]
        choices.append(choice)
    return tuple(choices)


# Blocks are sampled independently, so lr/dropout/weight decay and the four
# source-method mechanisms are not locked into inseparable profiles.  All
# method weights stay meaningfully positive; this pass never searches ERM-like
# configurations that silently switch EMBER modules off.
SPACES = {
    "bailA": {
        "architecture": architectures(
            ("GCN", 64, 1), ("GCN", 64, 2),
            ("GCN", 128, 2), ("GCN", 128, 3),
        ),
        "lr": values("lr", (0.003, 0.004, 0.006)),
        "weight_decay": values("lr2_reg", (1e-4, 3e-4, 1e-3)),
        "dropout": values("dropout", (0.4, 0.5, 0.6)),
        "fairness": values("lambda_fair", (4.0, 8.0, 16.0)),
        "meta_lr": values("meta_lr", (0.012, 0.016, 0.020)),
        "coordination": values("lambda_coord", (0.75, 1.0)),
        "sensitive": values("lambda_sen", (0.5, 1.0, 2.0)),
        "disentanglement": values("lambda_dec", (1.0, 2.0, 4.0, 8.0)),
        "mmd_scale": values("_mmd_at_64", (0.5, 0.70710678, 1.0)),
    },
    "germanA": {
        "architecture": architectures(
            ("GCN", 64, 1), ("GCN", 64, 2),
            ("GCN", 128, 2), ("GCN", 128, 3),
        ),
        "lr": values("lr", (0.003, 0.0045, 0.005, 0.006)),
        "weight_decay": values("lr2_reg", (1e-4, 3e-4, 1e-3)),
        "dropout": values("dropout", (0.3, 0.4, 0.5)),
        "fairness": values("lambda_fair", (2.0, 4.0, 8.0)),
        "meta_lr": values("meta_lr", (0.012, 0.014, 0.016)),
        "coordination": values("lambda_coord", (0.75, 1.0)),
        "sensitive": values("lambda_sen", (1.0, 2.0, 4.0)),
        "disentanglement": values("lambda_dec", (2.0, 4.0, 8.0)),
        "mmd_scale": values("_mmd_at_64", (0.5, 0.70710678, 1.0)),
    },
    "pokec": {
        # 128 x 4 is intentionally excluded for memory safety.
        "architecture": architectures(
            ("GCN", 64, 2), ("GCN", 64, 4), ("GCN", 128, 2),
        ),
        "lr": values("lr", (0.001, 0.0015, 0.002, 0.003)),
        "weight_decay": values("lr2_reg", (1e-4, 3e-4, 1e-3)),
        "dropout": values("dropout", (0.3, 0.4, 0.5)),
        "fairness": values("lambda_fair", (4.0, 8.0, 16.0)),
        "meta_lr": values("meta_lr", (0.016, 0.020)),
        "coordination": values("lambda_coord", (0.75, 1.0)),
        "sensitive": values("lambda_sen", (0.5, 1.0, 2.0)),
        "disentanglement": values("lambda_dec", (0.5, 1.0, 2.0, 8.0)),
        "mmd_scale": values("_mmd_at_64", (1.5, 2.0)),
    },
    "syn": {
        # The MLP backbone has two fixed linear layers; n_layers is irrelevant.
        "architecture": architectures(
            ("MLP", 128), ("MLP", 256), ("MLP", 512),
        ),
        "lr": values("lr", (0.003, 0.004, 0.005, 0.006)),
        "weight_decay": values("lr2_reg", (1e-4, 3e-4, 1e-3)),
        "dropout": values("dropout", (0.2, 0.3, 0.4, 0.5)),
        "fairness": values("lambda_fair", (4.0, 8.0, 16.0)),
        "meta_lr": values("meta_lr", (0.006, 0.008, 0.012, 0.016)),
        "coordination": values("lambda_coord", (0.75, 1.0)),
        "sensitive": values("lambda_sen", (0.5, 1.0, 2.0)),
        "disentanglement": values("lambda_dec", (4.0, 8.0, 16.0, 32.0)),
        "mmd_scale": values("_mmd_at_64", (0.5, 0.75, 1.0)),
    },
}


@dataclass(frozen=True)
class Trial:
    dataset: str
    number: int
    parameters: dict[str, Any]

    @property
    def key(self) -> str:
        payload = json.dumps(
            [SEARCH_VERSION, self.dataset, {**FIXED, **self.parameters}],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


@dataclass(frozen=True)
class Task:
    trial: Trial
    seed: int


PRINT_LOCK = threading.Lock()
FAILURE_LOCK = threading.Lock()


def status(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def shuffled_cycle(choices: Sequence[dict[str, Any]], seed: int):
    cycle = 0
    while True:
        order = list(range(len(choices)))
        random.Random(seed + 1000003 * cycle).shuffle(order)
        for index in order:
            yield choices[index]
        cycle += 1


def materialize(raw: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(raw)
    hidden = int(parameters["hidden_dim"])
    base_bandwidth = float(parameters.pop("_mmd_at_64"))
    parameters["source_mmd_bandwidth"] = float(
        f"{base_bandwidth * math.sqrt(hidden / 64.0):.8g}"
    )
    return parameters


def valid(dataset: str, parameters: Mapping[str, Any]) -> bool:
    return (
        int(parameters["hidden_dim"]) > 0
        and float(parameters["lr"]) > 0.0
        and float(parameters["lr2_reg"]) >= 0.0
        and 0.0 <= float(parameters["dropout"]) < 1.0
        and float(parameters["lambda_fair"]) > 0.0
        and float(parameters["lambda_sen"]) > 0.0
        and float(parameters["lambda_dec"]) > 0.0
        and 0.0 < float(parameters["lambda_coord"]) <= 1.0
        and float(parameters["meta_lr"]) * float(parameters["lambda_coord"])
        >= float(DATASETS[dataset]["min_meta_step"])
        and float(parameters["source_mmd_bandwidth"]) > 0.0
    )


def baseline_for(dataset: str) -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as source:
        configured = yaml.safe_load(source)[dataset]
    keys = {
        key
        for choices in SPACES[dataset].values()
        for choice in choices
        for key in choice
        if not key.startswith("_")
    }
    baseline = {key: configured[key] for key in keys if key in configured}
    baseline["source_mmd_bandwidth"] = float(configured["source_mmd_bandwidth"])
    return baseline


def make_trials(dataset: str, count: int, sampler_seed: int) -> list[Trial]:
    space = SPACES[dataset]
    streams = {
        block: shuffled_cycle(
            choices,
            stable_seed(sampler_seed, dataset, block),
        )
        for block, choices in space.items()
    }
    trials: list[Trial] = []
    signatures: set[str] = set()

    def add(parameters: Mapping[str, Any]) -> None:
        parameters = dict(parameters)
        signature = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        if valid(dataset, parameters) and signature not in signatures:
            signatures.add(signature)
            trials.append(Trial(dataset, len(trials), parameters))

    add(baseline_for(dataset))
    attempts = 0
    while len(trials) < count:
        raw: dict[str, Any] = {}
        for block in space:
            raw.update(next(streams[block]))
        add(materialize(raw))
        attempts += 1
        if attempts > max(10000, count * 1000):
            raise RuntimeError(f"Could not generate {count} unique {dataset} trials")
    return trials


def code_fingerprint() -> str:
    files = [
        Path(__file__),
        MAIN,
        ROOT / "runner.py",
        ROOT / "adaptation.py",
        ROOT / "dataset.py",
        ROOT / "learn.py",
        ROOT / "utils.py",
        ROOT / "config.py",
        ROOT / "models" / "__init__.py",
        ROOT / "models" / "classifier.py",
        ROOT / "models" / "encoder.py",
        CONFIG,
    ]
    digest = hashlib.sha256(SEARCH_VERSION.encode())
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def task_paths(results_dir: Path, task: Task) -> tuple[Path, Path, Path]:
    base = results_dir / task.trial.dataset / f"trial_{task.trial.key}"
    return (
        base / f"seed_{task.seed}.json",
        base / f"seed_{task.seed}.log",
        base / f"seed_{task.seed}.stderr.log",
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def source_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = payload["metrics"]["source_val"]
    result: dict[str, float] = {}
    for name in ("acc", "auc", "dp", "eo"):
        values = metrics[name]
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"source_val.{name} must contain exactly one run")
        result[name] = float(values[0])
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("source validation metrics must be finite")
    return result


def result_complete(path: Path, task: Task, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
        source_metrics(payload)
        tuning = payload["tuning"]
        return (
            payload.get("stage") == "source"
            and payload.get("dataset") == task.trial.dataset
            and int(payload.get("seed")) == task.seed
            and tuning.get("search_version") == SEARCH_VERSION
            and tuning.get("fingerprint") == fingerprint
            and tuning.get("trial_key") == task.trial.key
            and tuning.get("parameters") == {**FIXED, **task.trial.parameters}
        )
    except (IndexError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def override(key: str, value: Any) -> str:
    return f"{key}={value}"


def build_command(args: argparse.Namespace, gpu: str, task: Task) -> list[str]:
    raw, log, _ = task_paths(args.results_dir, task)
    inid, outid = DATASETS[task.trial.dataset]["domains"]
    command = [
        sys.executable,
        str(MAIN),
        "--dataset", task.trial.dataset,
        "--inid", inid,
        "--outid", outid,
        "--device_id", "0",
        "--seed", str(task.seed),
        "--runs_override", "1",
        "--ablation", "full",
        "--source-only",
        "--no-checkpoint",
        "--disable-checkpoint-save",
        "--disable_embedding_export",
        "--log_path", str(log),
        "--result_path", str(raw),
    ]
    parameters = {**FIXED, **task.trial.parameters}
    for key, value in parameters.items():
        command.extend(("--override", override(key, value)))
    return command


def gpu_state(selector: str) -> tuple[int, int]:
    command = [
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    line = completed.stdout.strip().splitlines()[0]
    free_memory, utilization = [int(part.strip()) for part in line.split(",")[:2]]
    return free_memory, utilization


def wait_for_gpu(args: argparse.Namespace, gpu: str, dataset: str) -> None:
    required = max(
        int(args.min_free_mb),
        int(DATASETS[dataset]["min_free_mb"]),
    )
    while True:
        free_memory, utilization = gpu_state(gpu)
        if free_memory >= required and utilization <= args.max_gpu_util:
            return
        status(
            f"[GPU {gpu}] waiting for {dataset}: free={free_memory} MiB "
            f"(need {required}), util={utilization}%"
        )
        time.sleep(args.gpu_poll_seconds)


def run_task(
    args: argparse.Namespace,
    gpu: str,
    task: Task,
    fingerprint: str,
) -> None:
    raw, log, stderr_path = task_paths(args.results_dir, task)
    if result_complete(raw, task, fingerprint):
        status(
            f"[GPU {gpu}] skip {task.trial.dataset} "
            f"trial={task.trial.key} seed={task.seed}"
        )
        return

    raw.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw.with_suffix(raw.suffix + ".tmp")
    raw.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    wait_for_gpu(args, gpu, task.trial.dataset)
    command = build_command(args, gpu, task)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    status(
        f"[GPU {gpu}] run {task.trial.dataset} "
        f"trial={task.trial.key} seed={task.seed}"
    )
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = time.monotonic() - started
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{task.trial.dataset} trial={task.trial.key} seed={task.seed} "
            f"failed with exit code {completed.returncode}; see {stderr_path}"
        )
    if not raw.exists():
        raise RuntimeError(f"Missing result file: {raw}")

    payload = load_json(raw)
    source_metrics(payload)
    if (
        payload.get("stage") != "source"
        or payload.get("dataset") != task.trial.dataset
        or int(payload.get("seed")) != task.seed
    ):
        raise RuntimeError(f"Unexpected result metadata in {raw}")
    payload["tuning"] = {
        "search_version": SEARCH_VERSION,
        "fingerprint": fingerprint,
        "trial_key": task.trial.key,
        "trial_number": task.trial.number,
        "parameters": {**FIXED, **task.trial.parameters},
        "duration_seconds": duration,
    }
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    temporary.replace(raw)
    if not result_complete(raw, task, fingerprint):
        raise RuntimeError(f"Result validation failed after writing {raw}")


def worker(
    args: argparse.Namespace,
    gpu: str,
    tasks: queue.Queue,
    fingerprint: str,
    failures: list[str],
    stop: threading.Event,
) -> None:
    while True:
        task = tasks.get()
        try:
            if task is None:
                return
            if stop.is_set():
                continue
            run_task(args, gpu, task, fingerprint)
        except Exception as error:
            with FAILURE_LOCK:
                failures.append(str(error))
            status(f"[GPU {gpu}] ERROR: {error}")
            if args.fail_fast:
                stop.set()
        finally:
            tasks.task_done()


def mean_std(items: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(items), statistics.pstdev(items)


def aggregate(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    fingerprint: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        tasks = [Task(trial, seed) for seed in args.seeds]
        paths = [task_paths(args.results_dir, task)[0] for task in tasks]
        if not all(
            result_complete(path, task, fingerprint)
            for path, task in zip(paths, tasks)
        ):
            continue
        payloads = [load_json(path) for path in paths]
        metrics = [source_metrics(payload) for payload in payloads]
        values_by_metric = {
            name: [item[name] for item in metrics]
            for name in ("acc", "auc", "dp", "eo")
        }
        means = {
            name: statistics.fmean(values)
            for name, values in values_by_metric.items()
        }
        scores = [
            item["acc"] + item["auc"] - item["dp"] - item["eo"]
            for item in metrics
        ]
        row: dict[str, Any] = {
            "dataset": trial.dataset,
            "trial_number": trial.number,
            "combination_id": trial.key,
            "score": means["acc"] + means["auc"] - means["dp"] - means["eo"],
            "score_std": statistics.pstdev(scores),
            "seeds": json.dumps(list(args.seeds)),
            "parameters": json.dumps(
                {**FIXED, **trial.parameters},
                sort_keys=True,
            ),
            "result_paths": json.dumps([str(path) for path in paths]),
            "duration_seconds": sum(
                float(payload["tuning"]["duration_seconds"])
                for payload in payloads
            ),
        }
        for name, values in values_by_metric.items():
            row[name], row[f"{name}_std"] = mean_std(values)
        row.update(trial.parameters)
        rows.append(row)

    for dataset in args.datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        selected.sort(key=lambda row: (-row["score"], row["trial_number"]))
        for rank, row in enumerate(selected, 1):
            row["rank"] = rank
    rows.sort(key=lambda row: (args.datasets.index(row["dataset"]), row["rank"]))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    core = [
        "dataset", "rank", "trial_number", "combination_id",
        "score", "score_std", "acc", "acc_std", "auc", "auc_std",
        "dp", "dp_std", "eo", "eo_std", "duration_seconds", "seeds",
        "parameters", "result_paths",
    ]
    parameter_keys = [
        key
        for key in SOURCE_ORDER
        if key not in FIXED and any(key in trial.parameters for trial in trials)
    ]
    with (args.results_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=core + parameter_keys)
        writer.writeheader()
        writer.writerows(rows)

    best_source: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for dataset in args.datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        if not selected:
            continue
        best_config = dict(FIXED)
        best_config.update(json.loads(selected[0]["parameters"]))
        best_source[dataset] = {
            key: best_config[key]
            for key in SOURCE_ORDER
            if key in best_config
        }
        candidates[dataset] = [
            {
                "rank": row["rank"],
                "score": float(row["score"]),
                "score_std": float(row["score_std"]),
                "config": {
                    **FIXED,
                    **json.loads(row["parameters"]),
                },
            }
            for row in selected[:3]
        ]

    with (args.results_dir / "best_source.yaml").open("w", encoding="utf-8") as output:
        yaml.safe_dump(best_source, output, sort_keys=False, allow_unicode=True)
    with (args.results_dir / "source_candidates.yaml").open(
        "w", encoding="utf-8"
    ) as output:
        yaml.safe_dump(candidates, output, sort_keys=False, allow_unicode=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune only EMBER source-stage hyperparameters."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="override the default number of source configurations per dataset",
    )
    parser.add_argument(
        "--seeds",
        nargs=2,
        type=int,
        default=list(DEFAULT_SEEDS),
        metavar=("SEED1", "SEED2"),
        help="exactly two distinct source-training seeds",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0", "1", "2", "3", "4", "5"],
        help="physical GPU indices or UUID selectors",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sampler-seed", type=int, default=2027)
    parser.add_argument("--min-free-mb", type=int, default=6000)
    parser.add_argument("--max-gpu-util", type=int, default=70)
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    args.results_dir = args.results_dir.expanduser()
    if not args.results_dir.is_absolute():
        args.results_dir = (Path.cwd() / args.results_dir).resolve()

    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct values")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain distinct selectors")
    if args.min_free_mb < 0 or not 0 <= args.max_gpu_util <= 100:
        parser.error("invalid GPU admission thresholds")
    if args.gpu_poll_seconds < 1:
        parser.error("--gpu-poll-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    fingerprint = code_fingerprint()
    trials_by_dataset = {
        dataset: make_trials(
            dataset,
            args.trials or int(DATASETS[dataset]["trials"]),
            args.sampler_seed,
        )
        for dataset in args.datasets
    }
    trials = [
        trials_by_dataset[dataset][index]
        for index in range(max(map(len, trials_by_dataset.values())))
        for dataset in args.datasets
        if index < len(trials_by_dataset[dataset])
    ]

    status(
        "Source-stage trials: "
        + ", ".join(
            f"{dataset}={len(trials_by_dataset[dataset])}"
            for dataset in args.datasets
        )
        + f"; seeds={list(args.seeds)}"
    )
    if args.dry_run:
        for trial in trials[: min(8, len(trials))]:
            task = Task(trial, args.seeds[0])
            command = build_command(args, args.gpus[0], task)
            status(
                f"{trial.dataset} {trial.key}: "
                + subprocess.list2cmdline(command)
            )
        return

    tasks: queue.Queue = queue.Queue()
    for trial in trials:
        for seed in args.seeds:
            tasks.put(Task(trial, seed))
    for _ in args.gpus:
        tasks.put(None)

    failures: list[str] = []
    stop = threading.Event()
    workers = [
        threading.Thread(
            target=worker,
            args=(args, gpu, tasks, fingerprint, failures, stop),
            daemon=True,
        )
        for gpu in args.gpus
    ]
    for thread in workers:
        thread.start()
    tasks.join()
    for thread in workers:
        thread.join()

    rows = aggregate(args, trials, fingerprint)
    for dataset in args.datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        if selected:
            best = selected[0]
            status(
                f"{dataset}: best score={best['score']:.3f} "
                f"+/- {best['score_std']:.3f} ({best['combination_id']})"
            )
    status(f"Saved source-stage results under {args.results_dir}")
    if failures:
        raise RuntimeError(f"{len(failures)} task(s) failed; inspect stderr logs")


if __name__ == "__main__":
    main()
