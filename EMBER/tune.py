"""Coarse module-aware tuning for the current EMBER implementation.

Each parameter combination is run in two independent subprocesses and ranked
by ``mean(ACC) + mean(AUC) - mean(DP) - mean(EO)``.  The search uses coupled
profiles so Meta-Align, dual-head separation, cumulative prototype averaging,
minority-aware residual learning, and BCA remain meaningfully active.

Examples::

    python tune.py --dry-run --trials 4 --gpus 0
    python tune.py --gpus 0 1 2 3 4 5
    python tune.py --datasets pokec --trials 64 --gpus 0 1 2 3 4 5

When ``CUDA_VISIBLE_DEVICES`` is set, values passed to ``--gpus`` are logical
indices into that visible list.  Every child is restricted to one selected
GPU and receives ``--device_id 0``; Pokec admission therefore checks exactly
the same physical GPU that PyTorch will use.
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
import shlex
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.py"
CONFIG = ROOT / "config" / "config.yaml"
RESULTS_DIR = ROOT / "tune_results_coarse"

SEARCH_VERSION = "module_profiles_coarse_v1"
DEFAULT_SEEDS = (1111, 2222)
TARGET_SEED_OFFSET = 100000

DATASETS = {
    "bailA": {"domains": ("_2", "_1"), "nodes": 7661, "trials": 48},
    "germanA": {"domains": ("_2", "_1"), "nodes": 384, "trials": 48},
    "pokec": {"domains": ("_z", "_n"), "nodes": 66569, "trials": 64},
    "syn": {"domains": ("-2", "-1"), "nodes": 5000, "trials": 48},
}

# Numerical approximations and runtime constants are fixed, not searched.
# The 1024 cap makes class-conditional MMD a fresh bounded random estimate on
# large graphs.  Using 0 would construct the full O(n^2) Pokec kernel graph in
# every one of 800 epochs and is not viable for a multi-trial search.
FIXED = {
    "train_epochs": 800,
    "disentangle_batch_size": 512,
    "source_mmd_min_samples": 1,
    "source_mmd_max_samples": 1024,
    "mmd_chunk_size": 256,
}

BLOCKS = (
    "architecture",
    "optimizer",
    "meta_align",
    "dual_head",
    "prototype_residual",
    "bca",
)


def profiles(keys: Sequence[str], rows: Iterable[Sequence[Any]]) -> tuple[dict[str, Any], ...]:
    keys = tuple(keys)
    result = []
    for row in rows:
        row = tuple(row)
        if len(row) != len(keys):
            raise ValueError(f"Bad profile row {row!r} for {keys!r}")
        result.append(dict(zip(keys, row)))
    return tuple(result)


# The residual budget is adapt_lr * inner_steps * sqrt(hidden_dim).  The group
# prior is a fraction of the average binary (class, sensitive-group) count.
RESIDUAL_KEYS = (
    "adapt_epochs",
    "residual_inner_steps",
    "_residual_budget",
    "lambda_residual_l2",
    "tau_c",
    "_group_ratio",
)
LONG_RESIDUAL = profiles(
    RESIDUAL_KEYS,
    (
        (75, 20, 0.20, 0.10, 0.65, 0.010),
        (50, 20, 0.16, 0.30, 0.70, 0.003),
        (100, 10, 0.35, 0.03, 0.55, 0.050),
        (100, 15, 0.25, 0.10, 0.60, 0.100),
    ),
)
SHORT_RESIDUAL = profiles(
    RESIDUAL_KEYS,
    (
        (25, 20, 0.20, 0.10, 0.65, 0.010),
        (20, 20, 0.16, 0.30, 0.70, 0.003),
        (50, 10, 0.35, 0.03, 0.55, 0.050),
        (50, 15, 0.25, 0.10, 0.60, 0.100),
    ),
)

# Temperature is coupled to eta and delta_p because eta * temperature is the
# prior's approximate scale relative to cosine likelihood.  All rows keep BCA
# visible (minimum 0.20).  Five representative profiles are sufficient for
# the preliminary pass; the second round can expand around the winning region.
BCA = profiles(
    (
        "proto_temp",
        "lambda_pi",
        "prior_confidence_threshold",
        "prior_discount",
        "_prior_ratio",
    ),
    (
        (0.60, 0.70, 0.65, 0.80, 0.010),
        (0.40, 0.50, 0.60, 0.95, 0.003),
        (0.50, 0.80, 0.70, 0.50, 0.010),
        (0.80, 1.00, 0.70, 0.80, 0.020),
        (1.00, 1.00, 0.55, 0.95, 0.050),
    ),
)

DUAL_HEAD = profiles(
    ("lambda_sen", "_lambda_dec_at_64"),
    ((1.0, 2.0), (0.5, 0.5), (2.0, 2.0), (1.0, 8.0)),
)


def make_space(
    architecture: tuple[dict[str, Any], ...],
    optimizer: tuple[dict[str, Any], ...],
    meta_align: tuple[dict[str, Any], ...],
    residual: tuple[dict[str, Any], ...],
) -> dict[str, tuple[dict[str, Any], ...]]:
    return {
        "architecture": architecture,
        "optimizer": optimizer,
        "meta_align": meta_align,
        "dual_head": DUAL_HEAD,
        "prototype_residual": residual,
        "bca": BCA,
    }


SPACES = {
    "bailA": make_space(
        profiles(
            ("inter_encoder", "hidden_dim", "n_layers"),
            (("GCN", 64, 1), ("GCN", 64, 2), ("GCN", 128, 2)),
        ),
        profiles(
            ("lr", "lr2_reg", "dropout"),
            ((0.006, 1e-3, 0.60), (0.002, 1e-4, 0.30),
             (0.004, 1e-4, 0.50), (0.004, 1e-3, 0.30)),
        ),
        profiles(
            ("lambda_fair", "meta_lr", "lambda_coord", "_mmd_at_64"),
            ((8.0, 0.010, 1.00, 0.50), (2.0, 0.010, 0.50, 0.25),
             (16.0, 0.020, 0.75, 1.00), (4.0, 0.020, 1.00, 1.00)),
        ),
        LONG_RESIDUAL,
    ),
    "germanA": make_space(
        profiles(
            ("inter_encoder", "hidden_dim", "n_layers"),
            (("GCN", 128, 3), ("GCN", 32, 1),
             ("GCN", 64, 1), ("GCN", 64, 2)),
        ),
        profiles(
            ("lr", "lr2_reg", "dropout"),
            ((0.005, 1e-3, 0.40), (0.001, 1e-4, 0.30),
             (0.003, 1e-4, 0.30), (0.006, 1e-3, 0.50)),
        ),
        profiles(
            ("lambda_fair", "meta_lr", "lambda_coord", "_mmd_at_64"),
            ((2.0, 0.006, 1.00, 0.50), (1.0, 0.006, 0.50, 0.50),
             (8.0, 0.012, 0.75, 1.00), (2.0, 0.016, 0.75, 0.50)),
        ),
        SHORT_RESIDUAL,
    ),
    "pokec": make_space(
        # 128 x 4 is deliberately excluded for Pokec memory safety.
        profiles(
            ("inter_encoder", "hidden_dim", "n_layers"),
            (("GCN", 64, 4), ("GCN", 32, 2),
             ("GCN", 64, 2), ("GCN", 128, 2)),
        ),
        profiles(
            ("lr", "lr2_reg", "dropout"),
            ((0.0020, 1e-4, 0.40), (0.0005, 1e-4, 0.10),
             (0.0010, 1e-4, 0.30), (0.0020, 1e-3, 0.50),
             (0.0040, 1e-4, 0.30)),
        ),
        profiles(
            ("lambda_fair", "meta_lr", "lambda_coord", "_mmd_at_64"),
            ((8.0, 0.010, 1.00, 1.00), (2.0, 0.010, 0.50, 0.50),
             (16.0, 0.020, 0.75, 2.00), (4.0, 0.020, 1.00, 2.00)),
        ),
        SHORT_RESIDUAL,
    ),
    "syn": make_space(
        # Syn's MLP is always two linear layers; n_layers would have no effect.
        profiles(
            ("inter_encoder", "hidden_dim"),
            (("MLP", 128), ("MLP", 64), ("MLP", 256)),
        ),
        profiles(
            ("lr", "lr2_reg", "dropout"),
            ((0.005, 1e-4, 0.40), (0.001, 1e-4, 0.10),
             (0.003, 1e-3, 0.10), (0.006, 1e-3, 0.50)),
        ),
        profiles(
            ("lambda_fair", "meta_lr", "lambda_coord", "_mmd_at_64"),
            ((8.0, 0.006, 1.00, 0.50), (2.0, 0.006, 0.50, 0.50),
             (16.0, 0.012, 0.75, 1.00), (4.0, 0.016, 0.75, 1.00)),
        ),
        LONG_RESIDUAL,
    ),
}


@dataclass(frozen=True)
class Trial:
    dataset: str
    number: int
    parameters: dict[str, Any]
    profile_indices: dict[str, int]

    @property
    def signature(self) -> str:
        return json.dumps(self.parameters, sort_keys=True, separators=(",", ":"))

    @property
    def key(self) -> str:
        value = f"{self.dataset}|{self.signature}".encode()
        return hashlib.sha256(value).hexdigest()[:12]

    @property
    def name(self) -> str:
        return f"trial_{self.key}"


@dataclass(frozen=True)
class RunTask:
    trial: Trial
    seed: int
    wait_started: float | None = None


@dataclass(frozen=True)
class Device:
    logical_id: int
    selector: str | None

    @property
    def label(self) -> str:
        if self.selector is None:
            return "CPU"
        return f"GPU logical={self.logical_id} physical={self.selector}"


PRINT_LOCK = threading.Lock()
FAILURE_LOCK = threading.Lock()


def status(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def rounded(value: float) -> float:
    return float(f"{float(value):.8g}")


def effective_rounds(discount: float, epochs: int) -> float:
    return (1.0 - discount ** epochs) / (1.0 - discount)


def materialize(dataset: str, raw_parameters: Mapping[str, Any]) -> dict[str, Any]:
    p = dict(raw_parameters)
    hidden = int(p["hidden_dim"])
    dimension_scale = math.sqrt(hidden / 64.0)

    p["source_mmd_bandwidth"] = rounded(p.pop("_mmd_at_64") * dimension_scale)
    p["lambda_dec"] = rounded(p.pop("_lambda_dec_at_64") * hidden / 64.0)
    p["adapt_lr"] = rounded(
        p.pop("_residual_budget")
        / (int(p["residual_inner_steps"]) * math.sqrt(hidden))
    )
    nodes = DATASETS[dataset]["nodes"]
    p["group_pseudocount"] = rounded(p.pop("_group_ratio") * nodes / 4.0)
    p["prior_pseudocount"] = rounded(
        p.pop("_prior_ratio")
        * nodes
        * effective_rounds(float(p["prior_discount"]), int(p["adapt_epochs"]))
    )
    return p


def valid(parameters: Mapping[str, Any]) -> bool:
    positive = (
        "lr", "lambda_fair", "meta_lr", "lambda_coord", "lambda_sen",
        "lambda_dec", "source_mmd_bandwidth", "adapt_lr",
        "lambda_residual_l2", "group_pseudocount", "proto_temp",
        "lambda_pi", "prior_pseudocount",
    )
    return (
        all(float(parameters[key]) > 0.0 for key in positive)
        and int(parameters["hidden_dim"]) > 0
        and int(parameters["adapt_epochs"]) >= 10
        and int(parameters["residual_inner_steps"]) > 0
        and 0.0 <= float(parameters["dropout"]) < 1.0
        and 0.5 < float(parameters["tau_c"]) < 1.0
        and 0.5 < float(parameters["prior_confidence_threshold"]) < 1.0
        and 0.0 < float(parameters["prior_discount"]) < 1.0
        and 0.0 < float(parameters["lambda_coord"]) <= 1.0
        and 0.0 < float(parameters["lambda_pi"]) <= 1.0
        and float(parameters["lambda_pi"]) * float(parameters["proto_temp"]) >= 0.20
    )


def stable_seed(*parts: Any) -> int:
    value = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def shuffled_cycle(size: int, seed: int) -> Iterator[int]:
    cycle = 0
    while True:
        order = list(range(size))
        random.Random(seed + cycle * 1000003).shuffle(order)
        yield from order
        cycle += 1


def validate_spaces() -> None:
    for dataset, space in SPACES.items():
        if tuple(space) != BLOCKS:
            raise ValueError(f"Bad block order for {dataset}")
        seen: set[str] = set()
        baseline: dict[str, Any] = {}
        for block in BLOCKS:
            choices = space[block]
            if not choices:
                raise ValueError(f"Empty profile block {dataset}/{block}")
            keys = set(choices[0])
            if any(set(choice) != keys for choice in choices):
                raise ValueError(f"Inconsistent profile keys in {dataset}/{block}")
            if seen & keys:
                raise ValueError(f"Duplicate parameter keys in {dataset}/{block}")
            seen |= keys
            baseline.update(choices[0])
        parameters = materialize(dataset, baseline)
        if not valid(parameters):
            raise ValueError(f"Invalid baseline for {dataset}")
        if dataset == "syn" and "n_layers" in parameters:
            raise ValueError("n_layers must not be searched for Syn's fixed MLP")


def make_trials(dataset: str, count: int, sampler_seed: int) -> list[Trial]:
    space = SPACES[dataset]
    streams = {
        block: shuffled_cycle(
            len(space[block]), stable_seed(sampler_seed, dataset, block)
        )
        for block in BLOCKS
    }
    trials: list[Trial] = []
    signatures: set[str] = set()

    def add(indices: Mapping[str, int]) -> None:
        raw: dict[str, Any] = {}
        for block in BLOCKS:
            raw.update(space[block][indices[block]])
        parameters = materialize(dataset, raw)
        signature = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        if valid(parameters) and signature not in signatures:
            signatures.add(signature)
            trials.append(Trial(dataset, len(trials), parameters, dict(indices)))

    add({block: 0 for block in BLOCKS})
    attempts = 0
    while len(trials) < count:
        add({block: next(streams[block]) for block in BLOCKS})
        attempts += 1
        if attempts > max(10000, count * 1000):
            raise RuntimeError(f"Could not make {count} unique {dataset} trials")
    return trials


def interleave(trials_by_dataset: Mapping[str, Sequence[Trial]]) -> list[Trial]:
    datasets = list(trials_by_dataset)
    longest = max(map(len, trials_by_dataset.values()), default=0)
    return [
        trials_by_dataset[dataset][index]
        for index in range(longest)
        for dataset in datasets
        if index < len(trials_by_dataset[dataset])
    ]


def training_files() -> list[Path]:
    files = [path for path in ROOT.glob("*.py") if path.name != "tune.py"]
    files.extend((ROOT / "models").glob("*.py"))
    files.append(CONFIG)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in training_files():
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def task_paths(results_dir: Path, task: RunTask) -> tuple[Path, Path, Path]:
    base = results_dir / task.trial.dataset / task.trial.name
    raw = base / f"seed_{task.seed}.json"
    log = base / f"seed_{task.seed}.log"
    stderr = base / f"seed_{task.seed}.stderr.log"
    return raw, log, stderr


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def metrics_valid(payload: Mapping[str, Any]) -> bool:
    try:
        metrics = payload["metrics"]["target_after"]
        return all(
            isinstance(metrics[name], list)
            and len(metrics[name]) == 1
            and math.isfinite(float(metrics[name][0]))
            for name in ("acc", "auc", "dp", "eo")
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def result_complete(path: Path, task: RunTask, fingerprint: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
        tuning = payload["tuning"]
        if not isinstance(tuning, Mapping):
            return False
        duration = float(tuning.get("duration_seconds", -1.0))
        if not math.isfinite(duration) or duration < 0.0:
            return False
        inid, outid = DATASETS[task.trial.dataset]["domains"]
        target_seed = task.seed + TARGET_SEED_OFFSET
        return (
            metrics_valid(payload)
            and payload.get("dataset") == task.trial.dataset
            and payload.get("inid") == inid
            and payload.get("outid") == outid
            and payload.get("seed") == task.seed
            and payload.get("target_seed") == target_seed
            and payload.get("ablation") == "full"
            and tuning.get("version") == SEARCH_VERSION
            and tuning.get("fingerprint") == fingerprint
            and tuning.get("combination_id") == task.trial.key
            and tuning.get("parameters") == task.trial.parameters
            and tuning.get("fixed") == FIXED
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        return False


def override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    return f"{key}={value}"


def build_command(args: argparse.Namespace, device: Device, task: RunTask) -> list[str]:
    raw, log, _ = task_paths(args.results_dir, task)
    inid, outid = DATASETS[task.trial.dataset]["domains"]
    command = [
        sys.executable, str(MAIN),
        "--dataset", task.trial.dataset,
        "--inid", inid,
        "--outid", outid,
        "--device_id", "0" if device.selector is not None else "-1",
        "--seed", str(task.seed),
        "--target_seed", str(task.seed + TARGET_SEED_OFFSET),
        "--runs_override", "1",
        "--ablation", "full",
        "--no-checkpoint",
        "--disable-checkpoint-save",
        "--disable_embedding_export",
        "--log_path", str(log),
        "--result_path", str(raw),
    ]
    values = dict(FIXED)
    values.update(task.trial.parameters)
    for key, value in values.items():
        command.extend(("--override", override(key, value)))
    return command


def child_environment(device: Device) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device.selector or ""
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def nvidia_query(selector: str) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=index,uuid,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        )
    except FileNotFoundError as error:
        raise RuntimeError("nvidia-smi is required for GPU tuning") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"nvidia-smi timed out for GPU {selector}") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"Cannot query GPU {selector}: {detail}") from error

    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one nvidia-smi row for {selector}, got {rows!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"Unexpected nvidia-smi row: {rows[0]!r}")
    try:
        index, free_mb, total_mb = int(fields[0]), int(fields[2]), int(fields[3])
    except ValueError as error:
        raise RuntimeError(f"Unexpected nvidia-smi row: {rows[0]!r}") from error
    return {
        "selector": selector,
        "physical_index": index,
        "uuid": fields[1],
        "free_mb": free_mb,
        "total_mb": total_mb,
    }


def detected_gpu_indices() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not auto-detect GPUs with nvidia-smi") from error
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def visible_selectors() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    selectors = [token.strip() for token in raw.split(",") if token.strip()]
    if not selectors or selectors == ["-1"] or selectors == ["NoDevFiles"]:
        return []
    return selectors


def resolve_devices(tokens: Sequence[str], validate: bool) -> list[Device]:
    if "auto" in tokens:
        if len(tokens) != 1:
            raise ValueError("--gpus auto cannot be mixed with explicit IDs")
        visible = visible_selectors()
        selectors = detected_gpu_indices() if visible is None else visible
        devices = [Device(index, selector) for index, selector in enumerate(selectors)]
    else:
        ids = [int(token) for token in tokens]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("--gpus requires distinct IDs")
        if -1 in ids:
            if ids != [-1]:
                raise ValueError("CPU -1 cannot be mixed with GPUs")
            return [Device(-1, None)]
        if any(index < 0 for index in ids):
            raise ValueError("GPU IDs must be non-negative, or -1 for CPU")
        visible = visible_selectors()
        if visible is not None:
            if any(index >= len(visible) for index in ids):
                raise ValueError(
                    f"GPU logical IDs {ids} exceed CUDA_VISIBLE_DEVICES={visible}"
                )
            devices = [Device(index, visible[index]) for index in ids]
        else:
            devices = [Device(index, str(index)) for index in ids]

    if not devices:
        raise ValueError("No visible GPU; use --gpus -1 for CPU")
    if len({device.selector for device in devices}) != len(devices):
        raise ValueError("Selected GPU entries map to duplicate physical devices")
    if validate:
        canonical_uuids = []
        for device in devices:
            canonical_uuids.append(nvidia_query(device.selector)["uuid"])
        if len(canonical_uuids) != len(set(canonical_uuids)):
            raise ValueError("Selected GPU entries resolve to the same physical GPU")
    return devices


def filter_pokec_devices(args: argparse.Namespace, devices: Sequence[Device]) -> list[Device]:
    if "pokec" not in args.datasets or devices[0].selector is None:
        return list(devices)
    eligible = []
    for device in devices:
        gpu = nvidia_query(device.selector)
        if gpu["total_mb"] >= args.pokec_min_free_mb:
            eligible.append(device)
        else:
            status(f"[skip] {device.label}: only {gpu['total_mb']} MiB total VRAM")
    if not eligible:
        raise RuntimeError(
            f"No selected GPU has {args.pokec_min_free_mb} MiB total VRAM"
        )
    return eligible


def pokec_admission(
    args: argparse.Namespace,
    device: Device,
    task: RunTask,
) -> tuple[dict[str, Any] | None, RunTask | None]:
    if device.selector is None:
        return None, None
    gpu = nvidia_query(device.selector)
    if gpu["free_mb"] >= args.pokec_min_free_mb:
        status(f"[{device.label}] Pokec admitted: {gpu['free_mb']} MiB free")
        return gpu, None

    now = time.monotonic()
    wait_started = task.wait_started if task.wait_started is not None else now
    if args.gpu_wait_timeout > 0 and now - wait_started >= args.gpu_wait_timeout:
        raise TimeoutError(
            f"No selected GPU admitted {task.trial.name} within "
            f"{args.gpu_wait_timeout:.0f}s"
        )
    status(
        f"[{device.label}] requeue Pokec: {gpu['free_mb']} MiB free, "
        f"need {args.pokec_min_free_mb} MiB"
    )
    return None, RunTask(task.trial, task.seed, wait_started)


def enrich_result(
    path: Path,
    task: RunTask,
    fingerprint: str,
    device: Device,
    duration: float,
    gpu_admission: Mapping[str, Any] | None,
) -> None:
    payload = load_json(path)
    payload["tuning"] = {
        "version": SEARCH_VERSION,
        "fingerprint": fingerprint,
        "combination_id": task.trial.key,
        "trial_number": task.trial.number,
        "parameters": task.trial.parameters,
        "fixed": FIXED,
        "profile_indices": task.trial.profile_indices,
        "seed": task.seed,
        "target_seed": task.seed + TARGET_SEED_OFFSET,
        "device": {"logical_id": device.logical_id, "selector": device.selector},
        "duration_seconds": duration,
    }
    if gpu_admission is not None:
        payload["tuning"]["gpu_admission"] = dict(gpu_admission)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    temporary.replace(path)


def run_task(
    args: argparse.Namespace,
    device: Device,
    task: RunTask,
    fingerprint: str,
    admission: Mapping[str, Any] | None,
) -> None:
    raw, log, stderr = task_paths(args.results_dir, task)
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not args.rerun and result_complete(raw, task, fingerprint):
        status(f"[resume] {task.trial.dataset} {task.trial.name} seed={task.seed}")
        return
    if raw.exists():
        raw.unlink()

    command = build_command(args, device, task)
    status(
        f"[{device.label}] start {task.trial.dataset} "
        f"{task.trial.name} seed={task.seed}"
    )
    started = time.monotonic()
    with stderr.open("w", encoding="utf-8") as error_log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=child_environment(device),
            stdout=subprocess.DEVNULL,
            stderr=error_log,
        )
    duration = time.monotonic() - started

    if completed.returncode != 0:
        raise RuntimeError(
            f"{task.trial.dataset} {task.trial.name} seed={task.seed} exited "
            f"with {completed.returncode}; see {log} and {stderr}"
        )
    if not raw.exists() or not metrics_valid(load_json(raw)):
        raise RuntimeError(f"Incomplete result for {task.trial.name} seed={task.seed}")
    enrich_result(raw, task, fingerprint, device, duration, admission)
    if not result_complete(raw, task, fingerprint):
        raise RuntimeError(f"Result metadata mismatch for {task.trial.name}")
    try:
        stderr.unlink()
    except OSError as error:
        status(f"[warning] could not remove {stderr}: {error}")
    status(
        f"[{device.label}] done  {task.trial.dataset} {task.trial.name} "
        f"seed={task.seed} ({duration / 60:.1f} min)"
    )


def worker(
    args: argparse.Namespace,
    device: Device,
    tasks: queue.Queue,
    failures: list[str],
    stop: threading.Event,
    shutdown: threading.Event,
    fingerprint: str,
) -> None:
    while not shutdown.is_set():
        try:
            task = tasks.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if stop.is_set():
                continue
            admission = None
            if task.trial.dataset == "pokec":
                admission, retry = pokec_admission(args, device, task)
                if retry is not None:
                    tasks.put(retry)
                    time.sleep(min(args.gpu_poll_seconds, 5))
                    continue
            run_task(args, device, task, fingerprint, admission)
        except Exception as error:
            with FAILURE_LOCK:
                failures.append(str(error))
            status(f"[{device.label}] ERROR: {error}")
            if args.fail_fast:
                stop.set()
        finally:
            tasks.task_done()


def metric(payload: Mapping[str, Any], name: str) -> float:
    return float(payload["metrics"]["target_after"][name][0])


CONFIG_ORDER = (
    "inter_encoder", "hidden_dim", "n_layers", "lr", "lr2_reg",
    "train_epochs", "dropout", "lambda_fair", "meta_lr", "lambda_coord",
    "lambda_sen", "lambda_dec", "disentangle_batch_size",
    "source_mmd_bandwidth", "source_mmd_min_samples",
    "source_mmd_max_samples", "mmd_chunk_size", "adapt_epochs", "tau_c",
    "proto_temp", "adapt_lr", "residual_inner_steps", "lambda_residual_l2",
    "group_pseudocount", "lambda_pi", "prior_confidence_threshold",
    "prior_pseudocount", "prior_discount",
)


def aggregate(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    fingerprint: str,
) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        tasks = [RunTask(trial, seed) for seed in args.seeds]
        paths = [task_paths(args.results_dir, task)[0] for task in tasks]
        if not all(
            result_complete(path, task, fingerprint)
            for path, task in zip(paths, tasks)
        ):
            continue
        payloads = [load_json(path) for path in paths]
        values = {
            name: [metric(payload, name) for payload in payloads]
            for name in ("acc", "auc", "dp", "eo")
        }
        means = {name: statistics.fmean(value) for name, value in values.items()}
        scores = [
            values["acc"][i] + values["auc"][i]
            - values["dp"][i] - values["eo"][i]
            for i in range(len(args.seeds))
        ]
        row = {
            "dataset": trial.dataset,
            "trial_number": trial.number,
            "combination_id": trial.key,
            "score": means["acc"] + means["auc"] - means["dp"] - means["eo"],
            "score_std": statistics.pstdev(scores),
            "duration_seconds": sum(
                float(payload["tuning"]["duration_seconds"]) for payload in payloads
            ),
            "seeds": json.dumps(args.seeds),
            "profile_indices": json.dumps(trial.profile_indices, sort_keys=True),
            "result_paths": json.dumps([str(path) for path in paths]),
            "parameters": json.dumps(trial.parameters, sort_keys=True),
        }
        for name in ("acc", "auc", "dp", "eo"):
            row[name] = means[name]
            row[f"{name}_std"] = statistics.pstdev(values[name])
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
        "dataset", "rank", "trial_number", "combination_id", "score",
        "score_std", "acc", "acc_std", "auc", "auc_std", "dp", "dp_std",
        "eo", "eo_std", "duration_seconds", "seeds", "profile_indices",
        "parameters", "result_paths",
    ]
    parameter_keys = sorted({key for trial in trials for key in trial.parameters})
    with (args.results_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=core + parameter_keys)
        writer.writeheader()
        writer.writerows(rows)

    with (args.results_dir / "best_config.yaml").open("w", encoding="utf-8") as output:
        for dataset in args.datasets:
            best = next(
                (row for row in rows if row["dataset"] == dataset and row["rank"] == 1),
                None,
            )
            if best is None:
                continue
            config = dict(FIXED)
            config.update(json.loads(best["parameters"]))
            output.write(f"{dataset}:\n")
            for key in CONFIG_ORDER:
                if key in config:
                    output.write(f"  {key}: {json.dumps(config[key])}\n")
    return rows


def write_manifest(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    devices: Sequence[Device],
    fingerprint: str,
) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SEARCH_VERSION,
        "fingerprint": fingerprint,
        "objective": "mean_acc + mean_auc - mean_dp - mean_eo",
        "datasets": args.datasets,
        "trial_counts": args.trial_counts,
        "seeds": args.seeds,
        "fixed": FIXED,
        "devices": [device.__dict__ for device in devices],
        "trials": [
            {
                "dataset": trial.dataset,
                "number": trial.number,
                "combination_id": trial.key,
                "profile_indices": trial.profile_indices,
                "parameters": trial.parameters,
            }
            for trial in trials
        ],
    }
    with (args.results_dir / "manifest.json").open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)


def trial_counts(args: argparse.Namespace) -> dict[str, int]:
    counts = {dataset: DATASETS[dataset]["trials"] for dataset in args.datasets}
    if args.trials is not None:
        if args.trials < 1:
            raise ValueError("--trials must be positive")
        return {dataset: args.trials for dataset in args.datasets}
    seen = set()
    for item in args.trials_per_dataset or ():
        dataset, separator, raw_count = item.partition("=")
        if not separator or dataset not in args.datasets or dataset in seen:
            raise ValueError(f"Invalid dataset trial budget: {item!r}")
        count = int(raw_count)
        if count < 1:
            raise ValueError(f"Trial budget must be positive: {item!r}")
        counts[dataset] = count
        seen.add(dataset)
    return counts


def dry_run(
    args: argparse.Namespace,
    trials_by_dataset: Mapping[str, Sequence[Trial]],
    trials: Sequence[Trial],
    devices: Sequence[Device],
    fingerprint: str,
) -> None:
    status(
        f"Dry run: {len(trials)} combinations, {len(trials) * len(args.seeds)} "
        f"independent runs, devices={[device.label for device in devices]}, "
        f"code={fingerprint[:12]}"
    )
    status(f"  budgets={args.trial_counts}; seeds={args.seeds}")
    for dataset, dataset_trials in trials_by_dataset.items():
        coverage = {
            block: dict(sorted(Counter(
                trial.profile_indices[block] for trial in dataset_trials
            ).items()))
            for block in BLOCKS
        }
        status(f"  {dataset}: {len(dataset_trials)} combinations; coverage={coverage}")
        status(f"    baseline={dataset_trials[0].parameters}")
    if trials:
        task = RunTask(trials[0], args.seeds[0])
        status(f"Example child CUDA_VISIBLE_DEVICES={devices[0].selector or ''}")
        status(shlex.join(build_command(args, devices[0], task)))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Tune full EMBER with two independent runs per combination."
    )
    result.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    budgets = result.add_mutually_exclusive_group()
    budgets.add_argument("--trials", type=int)
    budgets.add_argument(
        "--trials-per-dataset", nargs="+", metavar="DATASET=COUNT"
    )
    result.add_argument(
        "--gpus", nargs="+", default=["0"],
        help="logical GPU IDs, auto, or -1 for CPU",
    )
    result.add_argument(
        "--seeds", type=int, nargs=2, default=list(DEFAULT_SEEDS),
        metavar=("SEED1", "SEED2"),
    )
    result.add_argument("--sampler-seed", type=int, default=2027)
    result.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    result.add_argument("--pokec-min-free-mb", type=int, default=20000)
    result.add_argument("--gpu-poll-seconds", type=int, default=30)
    result.add_argument(
        "--gpu-wait-timeout", type=float, default=0.0,
        help="maximum Pokec wait in seconds; 0 waits indefinitely",
    )
    result.add_argument("--rerun", action="store_true")
    result.add_argument("--fail-fast", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("--datasets must not contain duplicates")
    if len(args.seeds) != 2 or len(set(args.seeds)) != 2:
        raise ValueError("--seeds requires exactly two distinct values")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative")
    if args.pokec_min_free_mb < 1 or args.gpu_poll_seconds < 1:
        raise ValueError("GPU memory threshold and poll interval must be positive")
    if args.gpu_wait_timeout < 0:
        raise ValueError("--gpu-wait-timeout must be non-negative")
    if not MAIN.exists() or not CONFIG.exists():
        raise FileNotFoundError("EMBER main.py or config/config.yaml is missing")


def main() -> None:
    arg_parser = parser()
    args = arg_parser.parse_args()
    args.results_dir = args.results_dir.resolve()
    try:
        args.trial_counts = trial_counts(args)
        args.seeds = tuple(args.seeds)
        validate_spaces()
        validate_args(args)
        devices = resolve_devices(args.gpus, validate=not args.dry_run)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        arg_parser.error(str(error))

    fingerprint = code_fingerprint()
    trials_by_dataset = {
        dataset: make_trials(dataset, args.trial_counts[dataset], args.sampler_seed)
        for dataset in args.datasets
    }
    trials = interleave(trials_by_dataset)
    if args.dry_run:
        dry_run(args, trials_by_dataset, trials, devices, fingerprint)
        return

    devices = filter_pokec_devices(args, devices)
    write_manifest(args, trials, devices, fingerprint)
    tasks: queue.Queue = queue.Queue()
    pending = 0
    for trial in trials:
        for seed in args.seeds:
            task = RunTask(trial, seed)
            raw = task_paths(args.results_dir, task)[0]
            if args.rerun or not result_complete(raw, task, fingerprint):
                tasks.put(task)
                pending += 1
    status(
        f"Launching {pending} pending seed tasks ({len(trials)} combinations) "
        f"on {[device.label for device in devices]}"
    )
    failures: list[str] = []
    stop = threading.Event()
    shutdown = threading.Event()
    workers = [
        threading.Thread(
            target=worker,
            args=(args, device, tasks, failures, stop, shutdown, fingerprint),
            daemon=True,
        )
        for device in devices
    ]
    for thread in workers:
        thread.start()
    tasks.join()
    shutdown.set()
    for thread in workers:
        thread.join()

    rows = aggregate(args, trials, fingerprint)
    status(f"Aggregated {len(rows)}/{len(trials)} complete combinations")
    for dataset in args.datasets:
        best = next(
            (row for row in rows if row["dataset"] == dataset and row["rank"] == 1),
            None,
        )
        if best:
            status(
                f"Best {dataset}: {best['combination_id']} "
                f"score={best['score']:.3f} +/- {best['score_std']:.3f}"
            )
    if failures:
        raise RuntimeError(
            f"{len(failures)} seed task(s) failed; completed results were retained"
        )


if __name__ == "__main__":
    main()
