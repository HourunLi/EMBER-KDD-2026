"""Tune EMBER once per hyperparameter combination, in parallel across GPUs.

The search space follows the method in Sections 1-3 of the EMBER paper:

* source utility/fairness coordination (beta, alpha, gamma and MMD scale),
* dual-head disentanglement temperature,
* confidence-filtered residual prototype evolution, and
* smoothed, discounted Bayesian class-prior correction.

This is a deterministic local-neighborhood plus balanced random search over
dataset-specific discrete ranges.  It deliberately runs exactly one training
run for every sampled combination (``--runs_override 1``).  Different trials are dispatched in
parallel with one worker per GPU, so a GPU never hosts two EMBER trials from
this script at the same time.  Before every Pokec launch, the worker also waits
for a configurable amount of free VRAM and low GPU utilization.

Examples
--------
Inspect the sampled trials without launching training::

    python tune.py --dry-run --trials 8

Use four GPUs for all four datasets::

    python tune.py --gpus 0 1 2 3 --trials 32

Tune only Pokec and resume already completed trials automatically::

    python tune.py --datasets pokec --gpus 0 1 --trials 48
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = SCRIPT_DIR / "main.py"
CONFIG_PATH = SCRIPT_DIR / "config" / "config.yaml"
SEARCH_ROUND = 2
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "tune_results_round2"

DATASET_DOMAINS: Dict[str, Tuple[str, str]] = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

# These shared values intentionally live in config.py instead of being
# repeated for every dataset in config/config.yaml.  They are listed here only
# so trial 0 can reproduce the complete baseline and so selected shared method
# parameters can still be searched explicitly.
SHARED_DEFAULTS: Dict[str, Any] = {
    "lambda_coord": 1.0,
    "disentangle_temp": 0.1,
    "group_pseudocount": 1.0,
    "prior_discount": 0.9,
}

# Second-round ranges are local neighborhoods around the best completed trial
# in paper/tune/summary.csv.  They retain a few adjacent values on either side
# of a boundary so the search can still discover a nearby improvement.  Bail
# and German use stricter confidence filtering; Pokec uses lower thresholds;
# the synthetic MLP setting permits a sharper prototype temperature.
SEARCH_SPACES: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    "bailA": {
        "hidden_dim": (64, 128),
        "n_layers": (1, 2),
        "lr": (0.003, 0.0045, 0.006),
        "dropout": (0.4, 0.5, 0.6),
        "lambda_fair": (2.0, 4.0, 6.0, 8.0),
        "meta_lr": (0.005, 0.01, 0.02, 0.03),
        "lambda_coord": (0.75, 1.0),
        "disentangle_temp": (0.03, 0.05, 0.1, 0.2),
        "source_mmd_bandwidth": (0.25, 0.5, 1.0, 1.5),
        "adapt_epochs": (75, 100, 125, 150),
        "adapt_lr": (0.0003, 0.0005, 0.0008),
        "residual_inner_steps": (20, 30, 40),
        "tau_c": (0.7, 0.75, 0.8),
        "prior_confidence_threshold": (0.6, 0.65, 0.7),
        "proto_temp": (0.5, 0.75, 1.0, 1.25),
        "lambda_pi": (0.0, 0.01, 0.05, 0.1, 0.25),
        "lambda_residual_l2": (0.0001, 0.001, 0.003),
        "group_pseudocount": (0.25, 0.5, 0.75, 1.0),
        "prior_pseudocount": (5.0, 10.0, 20.0),
        "prior_discount": (0.5, 0.75, 0.9, 0.95),
    },
    "germanA": {
        "hidden_dim": (32, 64, 128),
        "n_layers": (1, 2),
        "lr": (0.005, 0.0075, 0.01),
        "dropout": (0.2, 0.3, 0.4),
        "lambda_fair": (1.0, 2.0, 4.0, 8.0),
        "meta_lr": (0.01, 0.02, 0.03),
        "lambda_coord": (0.5, 0.75, 1.0),
        "disentangle_temp": (0.05, 0.1, 0.2),
        "source_mmd_bandwidth": (0.25, 0.5, 0.75, 1.5),
        "adapt_epochs": (10, 15, 25, 50),
        "adapt_lr": (0.0002, 0.0005, 0.001),
        "residual_inner_steps": (20, 40, 60),
        "tau_c": (0.6, 0.65, 0.7, 0.9),
        "prior_confidence_threshold": (0.6, 0.65, 0.7, 0.75),
        "proto_temp": (0.1, 0.15, 0.25, 0.5),
        "lambda_pi": (0.1, 0.4, 0.6, 0.8),
        "lambda_residual_l2": (0.001, 0.01, 0.03),
        "group_pseudocount": (1.0, 2.0, 5.0),
        "prior_pseudocount": (1.0, 5.0, 10.0),
        "prior_discount": (0.5, 0.75, 0.9),
    },
    "pokec": {
        "hidden_dim": (64, 128),
        "n_layers": (2, 3, 4),
        "lr": (0.001, 0.003, 0.004),
        "dropout": (0.1, 0.2, 0.3, 0.4),
        "lambda_fair": (0.5, 1.0, 2.0, 4.0),
        "meta_lr": (0.005, 0.01, 0.02),
        "lambda_coord": (0.5, 0.75, 1.0),
        "disentangle_temp": (0.05, 0.1, 0.2),
        "source_mmd_bandwidth": (0.5, 1.0, 2.0),
        "adapt_epochs": (20, 25, 50, 75),
        "adapt_lr": (0.0003, 0.0005, 0.001),
        "residual_inner_steps": (10, 20, 30),
        "tau_c": (0.2, 0.25, 0.35, 0.45),
        "prior_confidence_threshold": (0.2, 0.25, 0.35, 0.45),
        "proto_temp": (0.35, 0.5, 0.75, 1.0),
        "lambda_pi": (0.1, 0.2, 0.3, 0.4),
        "lambda_residual_l2": (0.0001, 0.001, 0.003),
        "group_pseudocount": (0.5, 1.0, 2.0, 5.0),
        "prior_pseudocount": (10.0, 50.0, 100.0),
        "prior_discount": (0.0, 0.25, 0.5, 0.9),
    },
    "syn": {
        # n_layers is intentionally absent: the MLP backbone ignores it.
        "hidden_dim": (128, 256),
        "lr": (0.002, 0.0035, 0.005),
        "dropout": (0.2, 0.3, 0.4),
        "lambda_fair": (2.0, 4.0, 8.0, 12.0),
        "meta_lr": (0.001, 0.003, 0.005, 0.01),
        "lambda_coord": (0.5, 0.75, 1.0),
        "disentangle_temp": (0.05, 0.1),
        "source_mmd_bandwidth": (0.25, 0.5, 1.0, 1.5),
        "adapt_epochs": (75, 100, 150),
        "adapt_lr": (0.0002, 0.0005, 0.001),
        "residual_inner_steps": (20, 40, 60),
        "tau_c": (0.35, 0.4, 0.45, 0.5),
        "prior_confidence_threshold": (0.35, 0.4, 0.45, 0.5),
        "proto_temp": (0.05, 0.1, 0.2, 0.5),
        "lambda_pi": (0.1, 0.5, 0.75, 1.0),
        "lambda_residual_l2": (0.0000003, 0.000001, 0.000003, 0.00001),
        "group_pseudocount": (0.5, 1.0, 2.0),
        "prior_pseudocount": (5.0, 10.0, 25.0, 50.0),
        "prior_discount": (0.75, 0.9, 0.95),
    },
}

# Keep the best completed first-round configuration as trial 0000.  This
# makes the second round monotone with respect to the observed baseline even
# if the new random samples all land below it.
SEARCH_ANCHORS: Dict[str, Dict[str, Any]] = {
    "bailA": {
        "hidden_dim": 64, "n_layers": 2, "lr": 0.003, "dropout": 0.6,
        "lambda_fair": 2.0, "meta_lr": 0.02, "lambda_coord": 1.0,
        "disentangle_temp": 0.2, "source_mmd_bandwidth": 0.25,
        "adapt_epochs": 150, "adapt_lr": 0.0003,
        "residual_inner_steps": 40, "tau_c": 0.75,
        "prior_confidence_threshold": 0.65, "proto_temp": 1.0,
        "lambda_pi": 0.01, "lambda_residual_l2": 0.001,
        "group_pseudocount": 1.0, "prior_pseudocount": 10.0,
        "prior_discount": 0.75,
    },
    "germanA": {
        "hidden_dim": 128, "n_layers": 2, "lr": 0.01, "dropout": 0.4,
        "lambda_fair": 8.0, "meta_lr": 0.02, "lambda_coord": 1.0,
        "disentangle_temp": 0.2, "source_mmd_bandwidth": 0.5,
        "adapt_epochs": 15, "adapt_lr": 0.0002,
        "residual_inner_steps": 40, "tau_c": 0.65,
        "prior_confidence_threshold": 0.75, "proto_temp": 0.25,
        "lambda_pi": 0.6, "lambda_residual_l2": 0.01,
        "group_pseudocount": 2.0, "prior_pseudocount": 1.0,
        "prior_discount": 0.75,
    },
    "pokec": {
        "hidden_dim": 128, "n_layers": 4, "lr": 0.004, "dropout": 0.2,
        "lambda_fair": 4.0, "meta_lr": 0.01, "lambda_coord": 0.5,
        "disentangle_temp": 0.2, "source_mmd_bandwidth": 0.5,
        "adapt_epochs": 25, "adapt_lr": 0.0005,
        "residual_inner_steps": 20, "tau_c": 0.25,
        "prior_confidence_threshold": 0.35, "proto_temp": 0.5,
        "lambda_pi": 0.2, "lambda_residual_l2": 0.001,
        "group_pseudocount": 0.5, "prior_pseudocount": 50.0,
        "prior_discount": 0.0,
    },
    "syn": {
        "hidden_dim": 256, "lr": 0.005, "dropout": 0.4,
        "lambda_fair": 8.0, "meta_lr": 0.01, "lambda_coord": 1.0,
        "disentangle_temp": 0.1, "source_mmd_bandwidth": 1.0,
        "adapt_epochs": 150, "adapt_lr": 0.0005,
        "residual_inner_steps": 40, "tau_c": 0.4,
        "prior_confidence_threshold": 0.4, "proto_temp": 0.1,
        "lambda_pi": 1.0, "lambda_residual_l2": 0.000001,
        "group_pseudocount": 1.0, "prior_pseudocount": 10.0,
        "prior_discount": 0.9,
    },
}

# The first local trials change one high-impact factor at a time around the
# anchor.  This is more informative than spending all 31 non-anchor trials on
# independent random combinations in a 20-dimensional space.
LOCAL_SEARCH_KEYS: Dict[str, Tuple[str, ...]] = {
    "bailA": (
        "hidden_dim", "tau_c", "prior_confidence_threshold", "adapt_epochs",
        "adapt_lr", "proto_temp", "lambda_fair", "meta_lr", "dropout",
        "lambda_residual_l2", "prior_discount", "source_mmd_bandwidth",
    ),
    "germanA": (
        "hidden_dim", "dropout", "tau_c", "prior_confidence_threshold",
        "proto_temp", "lambda_pi", "lambda_residual_l2", "adapt_epochs",
        "residual_inner_steps", "lambda_fair", "group_pseudocount",
        "source_mmd_bandwidth",
    ),
    "pokec": (
        "tau_c", "prior_confidence_threshold", "adapt_epochs", "adapt_lr",
        "proto_temp", "lambda_pi", "dropout", "hidden_dim",
        "n_layers", "lambda_residual_l2",
    ),
    "syn": (
        "hidden_dim", "tau_c", "prior_confidence_threshold", "prior_discount",
        "lambda_pi", "lambda_residual_l2", "source_mmd_bandwidth", "adapt_epochs",
        "dropout", "proto_temp",
    ),
}
LOCAL_TRIALS_PER_DATASET = 12


PRINT_LOCK = threading.Lock()
FAILURE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Trial:
    dataset: str
    trial_id: int
    parameters: Dict[str, Any]

    @property
    def name(self) -> str:
        return f"trial_{self.trial_id:04d}"


def print_status(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def load_base_config() -> Dict[str, Dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dataset mapping in {CONFIG_PATH}")
    return payload


def _stable_dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(dataset.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big")


def _signature(parameters: Mapping[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _balanced_parameter_columns(
    space: Mapping[str, Sequence[Any]],
    count: int,
    rng: random.Random,
) -> Dict[str, List[Any]]:
    """Give every categorical value near-equal representation in the sample."""
    columns: Dict[str, List[Any]] = {}
    for key, choices in space.items():
        if not choices:
            raise ValueError(f"Search range for {key} is empty")
        repeats = math.ceil(count / len(choices)) if count else 0
        values = list(choices) * repeats
        rng.shuffle(values)
        columns[key] = values[:count]
    return columns


def _local_anchor_candidates(
    dataset: str,
    space: Mapping[str, Sequence[Any]],
    anchor: Mapping[str, Any],
) -> Iterable[Dict[str, Any]]:
    """Yield one-factor perturbations in nearest-to-anchor order."""
    alternatives: Dict[str, List[Any]] = {}
    for key in LOCAL_SEARCH_KEYS[dataset]:
        anchor_value = anchor[key]
        values = [value for value in space[key] if value != anchor_value]
        try:
            values.sort(key=lambda value: abs(float(value) - float(anchor_value)))
        except (TypeError, ValueError):
            values.sort(key=repr)
        alternatives[key] = values

    max_depth = max((len(values) for values in alternatives.values()), default=0)
    for depth in range(max_depth):
        for key in LOCAL_SEARCH_KEYS[dataset]:
            values = alternatives[key]
            if depth >= len(values):
                continue
            candidate = dict(anchor)
            candidate[key] = values[depth]
            yield candidate


def make_trials(
    dataset: str,
    count: int,
    sampler_seed: int,
    base_config: Mapping[str, Mapping[str, Any]],
) -> List[Trial]:
    """Include the best first-round anchor, then draw balanced combinations."""
    space = SEARCH_SPACES[dataset]
    effective_base = dict(SHARED_DEFAULTS)
    effective_base.update(base_config[dataset])
    effective_base.update(SEARCH_ANCHORS[dataset])
    missing = [key for key in space if key not in effective_base]
    if missing:
        raise KeyError(
            f"No baseline/default value for {dataset}: {', '.join(sorted(missing))}"
        )

    baseline = {key: effective_base[key] for key in space}
    trials = [Trial(dataset=dataset, trial_id=0, parameters=baseline)]
    if count == 1:
        return trials

    seen = {_signature(baseline)}
    local_limit = min(count, 1 + LOCAL_TRIALS_PER_DATASET)
    for candidate in _local_anchor_candidates(dataset, space, baseline):
        if len(trials) >= local_limit:
            break
        signature = _signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(
            Trial(dataset=dataset, trial_id=len(trials), parameters=candidate)
        )

    remaining = count - len(trials)
    if remaining == 0:
        return trials

    rng = random.Random(_stable_dataset_seed(sampler_seed, dataset))
    columns = _balanced_parameter_columns(space, remaining, rng)
    attempts = 0
    index = 0
    while len(trials) < count:
        if index < remaining:
            parameters = {key: columns[key][index] for key in space}
            index += 1
        else:
            parameters = {key: rng.choice(choices) for key, choices in space.items()}
        signature = _signature(parameters)
        attempts += 1
        if signature in seen:
            if attempts > count * 100:
                raise RuntimeError(f"Could not sample {count} unique trials for {dataset}")
            continue
        seen.add(signature)
        trials.append(
            Trial(dataset=dataset, trial_id=len(trials), parameters=parameters)
        )
    return trials


def raw_result_path(results_dir: Path, trial: Trial) -> Path:
    return results_dir / "raw" / trial.dataset / f"{trial.name}.json"


def log_path(results_dir: Path, trial: Trial) -> Path:
    return results_dir / "logs" / trial.dataset / f"{trial.name}.log"


def stderr_path(results_dir: Path, trial: Trial) -> Path:
    return results_dir / "logs" / trial.dataset / f"{trial.name}.stderr.log"


def is_complete_result(
    path: Path,
    expected_parameters: Mapping[str, Any] | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as result_file:
            payload = json.load(result_file)
        target = payload["metrics"]["target_after"]
        for key in ("acc", "auc", "dp", "eo"):
            values = target.get(key)
            if not isinstance(values, list) or not values:
                return False
            if not math.isfinite(float(values[0])):
                return False
        if expected_parameters is not None:
            actual_parameters = payload.get("tuning", {}).get("parameters")
            if not isinstance(actual_parameters, dict):
                return False
            if _signature(actual_parameters) != _signature(expected_parameters):
                return False
        return True
    except (OSError, OverflowError, ValueError, KeyError, TypeError):
        return False


def _override_text(key: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def query_gpu_status() -> Dict[int, Dict[str, int]]:
    """Return current free/total memory and utilization for every GPU.

    The check intentionally uses ``nvidia-smi`` rather than importing torch:
    it runs before the child process initializes CUDA and therefore observes
    memory held by other users or stale jobs.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
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
            "nvidia-smi is required for Pokec GPU admission checks; "
            "use --gpus -1 for CPU or load the NVIDIA driver environment"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"nvidia-smi failed while checking GPU memory: {detail}") from error

    statuses: Dict[int, Dict[str, int]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            index, free_mb, total_mb, utilization = (int(field) for field in fields)
        except ValueError as error:
            raise RuntimeError(f"Unexpected nvidia-smi output: {line!r}") from error
        statuses[index] = {
            "free_mb": free_mb,
            "total_mb": total_mb,
            "utilization": utilization,
        }
    if not statuses:
        raise RuntimeError("nvidia-smi returned no GPU status rows")
    return statuses


def wait_for_pokec_gpu(
    args: argparse.Namespace,
    device_id: int,
) -> Dict[str, int] | None:
    """Wait until a Pokec worker has a safely idle GPU before launching.

    A 44-GB card with only a few hundred MB free can pass a superficial
    device-id check but still fail during the first backward pass.  Requiring
    a configurable free-memory reserve and low utilization prevents that
    situation and also avoids competing with another active process.
    """
    if device_id < 0:
        return None

    deadline = (
        time.monotonic() + args.gpu_wait_timeout
        if args.gpu_wait_timeout > 0
        else None
    )
    last_report = 0.0
    while True:
        status = query_gpu_status().get(device_id)
        if status is None:
            raise RuntimeError(f"GPU {device_id} was not reported by nvidia-smi")
        if status["total_mb"] < args.pokec_min_free_memory_mb:
            raise RuntimeError(
                f"GPU {device_id} has only {status['total_mb']} MiB total VRAM, "
                f"below the Pokec minimum-free threshold "
                f"{args.pokec_min_free_memory_mb} MiB"
            )
        if (
            status["free_mb"] >= args.pokec_min_free_memory_mb
            and status["utilization"] <= args.pokec_max_gpu_utilization
        ):
            print_status(
                f"[GPU {device_id}] Pokec admission granted: "
                f"free={status['free_mb']} MiB, util={status['utilization']}%"
            )
            return status

        now = time.monotonic()
        if now - last_report >= max(args.gpu_poll_seconds, 1):
            print_status(
                f"[GPU {device_id}] waiting for Pokec: "
                f"free={status['free_mb']} MiB (need >= "
                f"{args.pokec_min_free_memory_mb}), "
                f"util={status['utilization']}% (need <= "
                f"{args.pokec_max_gpu_utilization}%)"
            )
            last_report = now
        if deadline is not None and now >= deadline:
            raise TimeoutError(
                f"GPU {device_id} did not reach the Pokec memory threshold "
                f"within {args.gpu_wait_timeout:.0f}s"
            )
        time.sleep(args.gpu_poll_seconds)


def build_command(
    args: argparse.Namespace,
    device_id: int,
    trial: Trial,
    result_path: Path,
    trial_log_path: Path,
) -> List[str]:
    inid, outid = DATASET_DOMAINS[trial.dataset]
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--dataset", trial.dataset,
        "--inid", inid,
        "--outid", outid,
        "--device_id", str(device_id),
        "--seed", str(args.seed),
        "--target_seed", str(args.seed + args.target_seed_offset),
        "--runs_override", "1",
        "--no-checkpoint",
        "--disable-checkpoint-save",
        "--disable_embedding_export",
        "--log_path", str(trial_log_path),
        "--result_path", str(result_path),
    ]
    for key, value in trial.parameters.items():
        command.extend(("--override", _override_text(key, value)))
    return command


def _enrich_result(
    path: Path,
    trial: Trial,
    duration_seconds: float,
    device_id: int,
    gpu_status: Mapping[str, int] | None = None,
) -> None:
    with path.open("r", encoding="utf-8") as result_file:
        payload = json.load(result_file)
    payload["tuning"] = {
        "trial_id": trial.trial_id,
        "parameters": trial.parameters,
        "duration_seconds": duration_seconds,
        "device_id": device_id,
        "runs_per_combination": 1,
    }
    if gpu_status is not None:
        payload["tuning"]["gpu_admission"] = dict(gpu_status)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    temporary.replace(path)


def run_trial(
    args: argparse.Namespace,
    device_id: int,
    trial: Trial,
) -> None:
    result = raw_result_path(args.results_dir, trial)
    trial_log = log_path(args.results_dir, trial)
    trial_stderr = stderr_path(args.results_dir, trial)
    result.parent.mkdir(parents=True, exist_ok=True)
    trial_log.parent.mkdir(parents=True, exist_ok=True)

    if not args.rerun and is_complete_result(result, trial.parameters):
        print_status(f"[resume] {trial.dataset} {trial.name} already complete")
        return
    if args.rerun and result.exists():
        result.unlink()

    gpu_status = wait_for_pokec_gpu(args, device_id) if trial.dataset == "pokec" else None
    command = build_command(args, device_id, trial, result, trial_log)
    print_status(f"[device {device_id}] start {trial.dataset} {trial.name}")
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = time.monotonic() - started
    if completed.stderr:
        trial_stderr.write_text(completed.stderr, encoding="utf-8")
    elif trial_stderr.exists():
        trial_stderr.unlink()

    if completed.returncode != 0:
        raise RuntimeError(
            f"{trial.dataset} {trial.name} exited with {completed.returncode}; "
            f"see {trial_log} and {trial_stderr}"
        )
    if not is_complete_result(result):
        raise RuntimeError(f"{trial.dataset} {trial.name} did not produce {result}")
    _enrich_result(result, trial, duration, device_id, gpu_status)
    if not is_complete_result(result, trial.parameters):
        raise RuntimeError(f"{trial.dataset} {trial.name} result metadata mismatch")
    print_status(
        f"[device {device_id}] done  {trial.dataset} {trial.name} "
        f"({duration / 60.0:.1f} min)"
    )


def worker(
    args: argparse.Namespace,
    device_id: int,
    tasks: queue.Queue,
    failures: List[str],
    stop_event: threading.Event,
) -> None:
    while True:
        trial = tasks.get()
        try:
            if trial is None:
                return
            if stop_event.is_set():
                continue
            run_trial(args, device_id, trial)
        except Exception as error:  # keep other trials resumable by default
            message = str(error)
            with FAILURE_LOCK:
                failures.append(message)
            print_status(f"[device {device_id}] ERROR: {message}")
            if args.fail_fast:
                stop_event.set()
        finally:
            tasks.task_done()


def detect_gpus() -> List[int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    detected = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            detected.append(int(line))
    return detected


def resolve_devices(tokens: Sequence[str]) -> List[int]:
    if "auto" in tokens:
        if len(tokens) != 1:
            raise ValueError("--gpus auto cannot be combined with explicit GPU ids")
        devices = detect_gpus()
        return devices or [-1]

    devices = [int(token) for token in tokens]
    if not devices:
        raise ValueError("At least one GPU id is required; use -1 for CPU")
    if len(set(devices)) != len(devices):
        raise ValueError("--gpus must not contain duplicate ids")
    if any(device < -1 for device in devices):
        raise ValueError("GPU ids must be non-negative, or -1 for CPU")
    if -1 in devices and len(devices) != 1:
        raise ValueError("CPU id -1 cannot be combined with GPU ids")
    return devices


def filter_devices_for_pokec(
    args: argparse.Namespace,
    devices: Sequence[int],
) -> List[int]:
    """Exclude GPUs whose total VRAM cannot satisfy the Pokec reserve.

    A mixed cluster may contain a small GPU and a 44-GB GPU.  Keeping the
    small device in the worker pool would let it repeatedly pick Pokec jobs
    and fail before the larger device can consume them.  We therefore remove
    such devices for a run that includes Pokec; other datasets can still be
    tuned separately on the small device.
    """
    if "pokec" not in args.datasets or all(device_id < 0 for device_id in devices):
        return list(devices)

    statuses = query_gpu_status()
    eligible = [
        device_id
        for device_id in devices
        if device_id in statuses
        and statuses[device_id]["total_mb"] >= args.pokec_min_free_memory_mb
    ]
    excluded = [device_id for device_id in devices if device_id not in eligible]
    if excluded:
        print_status(
            "Excluding GPUs from this Pokec run because total VRAM is below "
            f"{args.pokec_min_free_memory_mb} MiB: {excluded}"
        )
    if not eligible:
        raise RuntimeError(
            "No selected GPU has enough total VRAM for Pokec; "
            f"need at least {args.pokec_min_free_memory_mb} MiB"
        )
    return eligible


def write_manifest(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    devices: Sequence[int],
) -> Path:
    path = args.results_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "search_round": SEARCH_ROUND,
        "search": "anchor_local_plus_balanced_random_categorical",
        "datasets": args.datasets,
        "trials_per_dataset": args.trials,
        "local_trials_per_dataset": min(args.trials, 1 + LOCAL_TRIALS_PER_DATASET),
        "runs_per_combination": 1,
        "training_seed": args.seed,
        "sampler_seed": args.sampler_seed,
        "devices": list(devices),
        "objective": {
            "formula": "ACC + AUC - DP - EO",
            "direction": "maximize",
        },
        "pokec_gpu_guard": {
            "min_free_memory_mb": args.pokec_min_free_memory_mb,
            "max_utilization_percent": args.pokec_max_gpu_utilization,
            "poll_seconds": args.gpu_poll_seconds,
            "wait_timeout_seconds": args.gpu_wait_timeout,
        },
        "search_spaces": {dataset: SEARCH_SPACES[dataset] for dataset in args.datasets},
        "trials": [
            {
                "dataset": trial.dataset,
                "trial_id": trial.trial_id,
                "parameters": trial.parameters,
            }
            for trial in trials
        ],
    }
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    return path


def _metric(payload: Mapping[str, Any], metric: str) -> float:
    values = payload["metrics"]["target_after"][metric]
    if not values:
        raise ValueError(f"Missing target-after metric: {metric}")
    return float(values[0])


def _pareto_flags(rows: Sequence[Mapping[str, Any]]) -> List[bool]:
    """Maximize ACC/AUC and minimize DP/EO simultaneously."""
    flags: List[bool] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                other["acc"] >= row["acc"]
                and other["auc"] >= row["auc"]
                and other["dp"] <= row["dp"]
                and other["eo"] <= row["eo"]
            )
            strictly_better = (
                other["acc"] > row["acc"]
                or other["auc"] > row["auc"]
                or other["dp"] < row["dp"]
                or other["eo"] < row["eo"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def aggregate(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    base_config: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for trial in trials:
        path = raw_result_path(args.results_dir, trial)
        if not is_complete_result(path, trial.parameters):
            continue
        with path.open("r", encoding="utf-8") as result_file:
            payload = json.load(result_file)
        acc = _metric(payload, "acc")
        auc = _metric(payload, "auc")
        dp = _metric(payload, "dp")
        eo = _metric(payload, "eo")
        score = acc + auc - dp - eo
        tuning = payload.get("tuning", {})
        rows.append(
            {
                "dataset": trial.dataset,
                "trial_id": trial.trial_id,
                "score": score,
                "acc": acc,
                "auc": auc,
                "dp": dp,
                "eo": eo,
                "duration_seconds": tuning.get("duration_seconds"),
                "parameters": trial.parameters,
                "result_path": str(path),
            }
        )

    best_configs: Dict[str, Dict[str, Any]] = {}
    for dataset in args.datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        dataset_rows.sort(key=lambda row: (-row["score"], row["trial_id"]))
        pareto = _pareto_flags(dataset_rows)
        for rank, (row, is_pareto) in enumerate(zip(dataset_rows, pareto), 1):
            row["rank"] = rank
            row["is_pareto"] = is_pareto
        if dataset_rows:
            best = dataset_rows[0]
            merged = dict(base_config[dataset])
            merged.update(best["parameters"])
            best_configs[dataset] = merged

    rows.sort(key=lambda row: (row["dataset"], row["rank"]))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "summary.json"
    csv_path = args.results_dir / "summary.csv"
    best_path = args.results_dir / "best_config.yaml"
    pareto_path = args.results_dir / "pareto_front.json"

    with json_path.open("w", encoding="utf-8") as output:
        json.dump(rows, output, ensure_ascii=False, indent=2)
    fieldnames = [
        "dataset", "rank", "trial_id", "score", "is_pareto",
        "acc", "auc", "dp", "eo", "duration_seconds",
        "parameters", "result_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["parameters"] = json.dumps(
                row["parameters"], sort_keys=True, ensure_ascii=False
            )
            writer.writerow(csv_row)
    with best_path.open("w", encoding="utf-8") as output:
        yaml.safe_dump(best_configs, output, sort_keys=False, allow_unicode=True)
    with pareto_path.open("w", encoding="utf-8") as output:
        json.dump(
            [row for row in rows if row["is_pareto"]],
            output,
            ensure_ascii=False,
            indent=2,
        )
    return rows, best_configs


def print_dry_run(
    args: argparse.Namespace,
    trials: Sequence[Trial],
    devices: Sequence[int],
) -> None:
    print_status(
        f"Dry run: {len(trials)} unique combinations, exactly one run each, "
        f"devices={list(devices)}"
    )
    for trial in trials:
        print_status(f"  {trial.dataset} {trial.name}: {trial.parameters}")
    if trials:
        example = trials[0]
        command = build_command(
            args,
            devices[0],
            example,
            raw_result_path(args.results_dir, example),
            log_path(args.results_dir, example),
        )
        print_status("Example command:")
        print_status(subprocess.list2cmdline(command))


def validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.pokec_min_free_memory_mb < 1:
        raise ValueError("--pokec-min-free-memory-mb must be positive")
    if not 0 <= args.pokec_max_gpu_utilization <= 100:
        raise ValueError("--pokec-max-gpu-utilization must lie in [0, 100]")
    if args.gpu_poll_seconds < 1:
        raise ValueError("--gpu-poll-seconds must be at least 1")
    if args.gpu_wait_timeout < 0:
        raise ValueError("--gpu-wait-timeout must be non-negative")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets must not contain duplicates")
    unknown = set(args.datasets) - set(DATASET_DOMAINS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing EMBER entry point: {MAIN_SCRIPT}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing EMBER config: {CONFIG_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one EMBER training run per sampled hyperparameter combination "
            "and parallelize trials across GPUs."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_DOMAINS),
        default=list(DATASET_DOMAINS),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=32,
        help="total unique combinations per dataset, including the baseline",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["auto"],
        metavar="ID",
        help="GPU ids (for example 0 1 2 3), auto, or -1 for CPU",
    )
    parser.add_argument(
        "--pokec-min-free-memory-mb",
        type=int,
        default=20000,
        help="minimum free VRAM required before starting a Pokec trial",
    )
    parser.add_argument(
        "--pokec-max-gpu-utilization",
        type=int,
        default=20,
        help="maximum GPU utilization allowed before starting Pokec",
    )
    parser.add_argument(
        "--gpu-poll-seconds",
        type=int,
        default=30,
        help="seconds between Pokec GPU admission checks",
    )
    parser.add_argument(
        "--gpu-wait-timeout",
        type=float,
        default=0.0,
        help="maximum wait in seconds; 0 waits indefinitely",
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--target-seed-offset", type=int, default=100000)
    parser.add_argument("--sampler-seed", type=int, default=2028)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild summaries from completed raw results without training",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.results_dir = args.results_dir.resolve()
    validate_args(args)
    devices = resolve_devices(args.gpus)
    if not args.dry_run and not args.aggregate_only:
        devices = filter_devices_for_pokec(args, devices)
    base_config = load_base_config()
    trials = [
        trial
        for dataset in args.datasets
        for trial in make_trials(
            dataset,
            args.trials,
            args.sampler_seed,
            base_config,
        )
    ]

    if args.dry_run:
        print_dry_run(args, trials, devices)
        return

    failures: List[str] = []
    if not args.aggregate_only:
        manifest = write_manifest(args, trials, devices)
        print_status(f"Saved tuning manifest to {manifest}")
        tasks: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        for trial in trials:
            if args.rerun or not is_complete_result(
                raw_result_path(args.results_dir, trial), trial.parameters
            ):
                tasks.put(trial)
        for _ in devices:
            tasks.put(None)

        workers = [
            threading.Thread(
                target=worker,
                args=(args, device_id, tasks, failures, stop_event),
                daemon=True,
            )
            for device_id in devices
        ]
        for thread in workers:
            thread.start()
        tasks.join()
        for thread in workers:
            thread.join()

    rows, best_configs = aggregate(args, trials, base_config)
    print_status(
        f"Aggregated {len(rows)}/{len(trials)} completed combinations under "
        f"{args.results_dir}"
    )
    for dataset, config in best_configs.items():
        best_row = next(
            row for row in rows if row["dataset"] == dataset and row["rank"] == 1
        )
        print_status(
            f"Best {dataset}: trial={best_row['trial_id']} "
            f"score={best_row['score']:.3f} config={config}"
        )
    if failures:
        raise RuntimeError(
            f"{len(failures)} trial(s) failed; successful trials and summaries were retained"
        )


if __name__ == "__main__":
    main()
