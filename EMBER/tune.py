"""Small single-dataset, two-stage grid-search launcher for EMBER."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import threading


# ============================= Edit only here =============================

DATASET = "bailA"                 # bailA | germanA | pokec | syn
STAGE = "source"                  # source | target
REPEATS = 3                       # repeated runs for every parameter combination
RANK_MODE = "acc+auc-dp-eo"       # acc+auc | acc+auc-dp-eo
GPUS = [0, 1]                     # one parallel worker per entry; use [-1] for CPU

BASE_SEED = 1111
TARGET_SEED = None                # None lets main.py derive per-run target seeds
RESUME = True
DRY_RUN = False
MAX_COMBINATIONS = 500            # 0 disables the safety limit
TOP_K = 20

# Source-stage search: source train split -> source validation metrics.
# A one-element list fixes that parameter. Expand only the ranges you want to tune.
SOURCE_SEARCH = {
    "cls_encoder": ["GCN"],
    "sens_encoder": ["GCN"],
    "hidden_dim": [64, 128],
    "cls_n_layers": [1, 2],
    "sens_n_layers": [1, 2],
    "lr": [0.002, 0.004],
    "lr2_reg": [0.0001],
    "train_epochs": [800],
    "dropout": [0.3, 0.5],
    "lambda_sen": [1.0],
    "lambda_dec": [1.0],
    "lambda_fair": [8.0],
    "meta_lr": [0.01],
    "lambda_coord": [0.75],
    "source_mmd_bandwidth": [1.0],
}

# Target-stage search: source settings come from config.yaml/checkpoints;
# ranking uses target-after-adaptation metrics.
TARGET_SEARCH = {
    "adapt_epochs": [50],
    "tau_c": [0.5, 0.7],
    "proto_temp": [0.5, 0.8],
    "adapt_lr": [0.001],
    "residual_inner_steps": [15],
    "lambda_residual_l2": [0.1],
    "group_pseudocount": [10.0],
    "lambda_pi": [0.5, 0.8],
    "prior_confidence_threshold": [0.65],
    "prior_pseudocount": [100.0],
    "prior_discount": [0.5, 0.8],
}

# Source/target domain identifiers used by main.py.
DOMAINS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

# ========================================================================


EMBER_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EMBER_DIR / "tuning" / DATASET / STAGE
PRINT_LOCK = threading.Lock()


def parameter_combinations(search_space):
    keys = list(search_space)
    values = [search_space[key] for key in keys]
    if any(not candidates for candidates in values):
        raise ValueError("Every search range must contain at least one value")
    return [dict(zip(keys, choice)) for choice in itertools.product(*values)]


def trial_id(parameters):
    config_bytes = (EMBER_DIR / "config" / "config.yaml").read_bytes()
    identity = {
        "dataset": DATASET,
        "stage": STAGE,
        "repeats": REPEATS,
        "base_seed": BASE_SEED,
        "target_seed": TARGET_SEED,
        "config_sha1": hashlib.sha1(config_bytes).hexdigest(),
        "parameters": parameters,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def trial_paths(identifier):
    return (
        OUTPUT_DIR / "raw" / f"{identifier}.json",
        OUTPUT_DIR / "logs" / f"{identifier}.log",
    )


def build_command(parameters, gpu, result_path, log_path):
    inid, outid = DOMAINS[DATASET]
    command = [
        sys.executable,
        str(EMBER_DIR / "main.py"),
        "--dataset", DATASET,
        "--inid", inid,
        "--outid", outid,
        "--device_id", str(gpu),
        "--runs_override", str(REPEATS),
        "--seed", str(BASE_SEED),
        "--result_path", str(result_path),
        "--log_path", str(log_path),
        "--disable_embedding_export",
    ]
    if TARGET_SEED is not None:
        command.extend(("--target_seed", str(TARGET_SEED)))
    if STAGE == "source":
        command.extend(("--source_only", "--no_checkpoint", "--disable_checkpoint_save"))
    for key, value in parameters.items():
        command.extend(("--override", f"{key}={value}"))
    return command


def valid_result(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric_key = "source_val" if STAGE == "source" else "target_after"
        values = payload["metrics"][metric_key]
        return all(name in values and len(values[name]) == REPEATS for name in ("acc", "auc", "dp", "eo"))
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return False


def run_trial(parameters, gpu, allow_resume=True):
    identifier = trial_id(parameters)
    result_path, log_path = trial_paths(identifier)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if RESUME and allow_resume and valid_result(result_path):
        with PRINT_LOCK:
            print(f"[skip] {identifier} already complete")
        return identifier, parameters, True, ""

    with PRINT_LOCK:
        print(f"[run ] {identifier} gpu={gpu} params={parameters}")
    process = subprocess.run(
        build_command(parameters, gpu, result_path, log_path),
        cwd=EMBER_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.stderr:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("\n[stderr]\n")
            stream.write(process.stderr)
    success = process.returncode == 0 and valid_result(result_path)
    with PRINT_LOCK:
        print(f"[{'done' if success else 'fail'}] {identifier}")
    return identifier, parameters, success, process.stderr.strip()


def run_parallel(trials):
    def run_gpu_queue(gpu, assigned_trials):
        return [run_trial(parameters, gpu) for parameters in assigned_trials]

    queues = [trials[index::len(GPUS)] for index in range(len(GPUS))]
    completed = []
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = {
            pool.submit(run_gpu_queue, gpu, assigned): gpu
            for gpu, assigned in zip(GPUS, queues)
            if assigned
        }
        for future in as_completed(futures):
            completed.extend(future.result())
    return completed


def finite_mean(values):
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        return float("nan")
    return sum(numbers) / len(numbers)


def summarize_trial(identifier, parameters):
    result_path, _ = trial_paths(identifier)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metric_key = "source_val" if STAGE == "source" else "target_after"
    metrics = payload["metrics"][metric_key]
    row = {name: finite_mean(metrics[name]) for name in ("acc", "auc", "dp", "eo")}
    row["acc+auc"] = row["acc"] + row["auc"]
    row["acc+auc-dp-eo"] = row["acc+auc"] - row["dp"] - row["eo"]
    row.update({"trial": identifier, "parameters": parameters})
    return row


def ranking_key(row):
    score = row[RANK_MODE]
    return score if math.isfinite(score) else -math.inf


def write_rankings(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.sort(key=ranking_key, reverse=True)

    json_rows = []
    for rank, row in enumerate(rows, start=1):
        record = {"rank": rank, **row}
        json_rows.append({
            key: (value if not isinstance(value, float) or math.isfinite(value) else None)
            for key, value in record.items()
        })
    (OUTPUT_DIR / "ranking.json").write_text(
        json.dumps(json_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    parameter_names = list((SOURCE_SEARCH if STAGE == "source" else TARGET_SEARCH))
    columns = ["rank", "trial", "acc", "auc", "dp", "eo", "acc+auc", "acc+auc-dp-eo", *parameter_names]
    with (OUTPUT_DIR / "ranking.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({
                "rank": rank,
                "trial": row["trial"],
                "acc": row["acc"],
                "auc": row["auc"],
                "dp": row["dp"],
                "eo": row["eo"],
                "acc+auc": row["acc+auc"],
                "acc+auc-dp-eo": row["acc+auc-dp-eo"],
                **row["parameters"],
            })
    return rows


def validate_settings():
    if DATASET not in DOMAINS:
        raise ValueError(f"Unknown DATASET: {DATASET}")
    if STAGE not in {"source", "target"}:
        raise ValueError("STAGE must be 'source' or 'target'")
    if RANK_MODE not in {"acc+auc", "acc+auc-dp-eo"}:
        raise ValueError("Unsupported RANK_MODE")
    if REPEATS < 1:
        raise ValueError("REPEATS must be positive")
    if not GPUS:
        raise ValueError("GPUS must contain at least one device id")


def main():
    validate_settings()
    search_space = SOURCE_SEARCH if STAGE == "source" else TARGET_SEARCH
    trials = parameter_combinations(search_space)
    if MAX_COMBINATIONS and len(trials) > MAX_COMBINATIONS:
        raise ValueError(
            f"Grid contains {len(trials)} combinations; increase MAX_COMBINATIONS "
            "or narrow the ranges"
        )
    print(
        f"dataset={DATASET} stage={STAGE} combinations={len(trials)} "
        f"repeats={REPEATS} rank={RANK_MODE} gpus={GPUS}"
    )

    if DRY_RUN:
        for parameters in trials:
            print(trial_id(parameters), parameters)
        return

    completed = []
    if STAGE == "target" and trials:
        # Prepare all source checkpoints once before target combinations start
        # in parallel, avoiding concurrent writes to the same checkpoint files.
        warmup = run_trial(trials[0], GPUS[0], allow_resume=False)
        if not warmup[2]:
            raise RuntimeError(f"Target checkpoint warm-up failed: {warmup[3]}")
        completed.append(warmup)
        trials = trials[1:]
    completed.extend(run_parallel(trials))

    successful = [(identifier, parameters) for identifier, parameters, ok, _ in completed if ok]
    failures = [
        {"trial": identifier, "parameters": parameters, "error": error}
        for identifier, parameters, ok, error in completed
        if not ok
    ]
    rows = write_rankings([
        summarize_trial(identifier, parameters)
        for identifier, parameters in successful
    ])
    (OUTPUT_DIR / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nTop {min(TOP_K, len(rows))} by {RANK_MODE}:")
    for rank, row in enumerate(rows[:TOP_K], start=1):
        print(
            f"{rank:>3}. score={row[RANK_MODE]:.4f} "
            f"acc={row['acc']:.4f} auc={row['auc']:.4f} "
            f"dp={row['dp']:.4f} eo={row['eo']:.4f} "
            f"params={row['parameters']}"
        )
    print(f"\nResults: {OUTPUT_DIR / 'ranking.csv'}")
    if failures:
        print(f"Failed combinations: {len(failures)} (see failures.json)")


if __name__ == "__main__":
    main()
