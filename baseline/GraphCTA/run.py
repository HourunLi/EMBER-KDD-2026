"""Run GraphCTA on the four EMBER domain pairs and aggregate five runs."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

from EMBER.dataset import get_dataset  # noqa: E402
from EMBER.utils import fair_metric  # noqa: E402
from visualization.export_utils import save_visualization_embeddings  # noqa: E402


PAIRS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}
# Edit this list to choose the GPUs used for parallel runs. One run per GPU.
AVAILABLE_GPUS = [0, 1, 2, 3]
SEEDS = [1111, 2222, 3333, 4444, 5555]
SOURCE_EPOCHS = TARGET_EPOCHS = 1000
PATIENCE = 100
# Edit these values directly. 0 disables the GPU-memory wait for that dataset.
DATASET_MIN_FREE_MIB = {
    "bailA": 0,
    "germanA": 0,
    "pokec": 0,
    "syn": 0,
}
GPU_POLL_SECONDS = 60

LOG_DIR = HERE / "logs"
RUN_DIR = HERE / "results" / "runs"
WORK_DIR = HERE / "results" / "work"
SUMMARY = HERE / "GraphCTA_5runs_summary.md"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def wait_for_dataset_gpu(dataset, gpu, min_free_mib, poll_seconds, stage):
    """Wait until the assigned GPU satisfies this dataset's free-memory floor."""
    if gpu < 0 or min_free_mib <= 0:
        return
    torch.cuda.set_device(gpu)
    while True:
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu)
        free_mib = free_bytes / 1024**2
        total_mib = total_bytes / 1024**2
        if total_mib + 1e-6 < min_free_mib:
            raise RuntimeError(
                f"{dataset} requires at least {min_free_mib:.0f} MiB free, but "
                f"GPU {gpu} has only {total_mib:.0f} MiB total memory."
            )
        print(
            f"[GPU guard] dataset={dataset} stage={stage} gpu={gpu} "
            f"free={free_mib:.0f}/{total_mib:.0f} MiB "
            f"required={min_free_mib:.0f} MiB",
            flush=True,
        )
        if free_mib >= min_free_mib:
            return
        print(
            f"[GPU guard] insufficient free memory; retrying in "
            f"{poll_seconds}s",
            flush=True,
        )
        time.sleep(poll_seconds)


def valid_mask(data):
    return (data.y == 0) | (data.y == 1)


def all_valid_mask(data):
    return (data.train_mask | data.val_mask | data.test_mask) & valid_mask(data)


def graphcta_view(data, is_target=False):
    """Keep the graph while preventing target labels from entering adaptation."""
    view = data.clone()
    if is_target:
        view.y.zero_()
        view.y[1] = 1  # GraphCTA only uses target y here to infer class count.
    else:
        view.y[~valid_mask(view)] = 0
    return view


def install_pair_dataset(source, target):
    """Feed EMBER Data objects through GraphCTA's unchanged dataset interface."""
    graphcta_datasets = importlib.import_module("datasets")

    class PairDataset:
        def __init__(self, _path, name):
            self.data = source if name == "Citationv1" else target

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.data.clone()

    graphcta_datasets.CitationDataset = PairDataset


def import_stage(name, argv):
    sys.modules.pop(name, None)
    sys.argv = [f"{name}.py", *argv]
    return importlib.import_module(name)


def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_source(device, seed, epochs, patience):
    seed_all(seed)
    module = import_stage(
        "train_source",
        [
            "--device", str(device),
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--patience", str(patience),
        ],
    )
    best_epoch = module.train_source()
    module.model.load_state_dict(
        torch.load(f"{best_epoch}.pth", map_location=device)
    )
    torch.save(cpu_state(module.model), "model.pth")
    Path(f"{best_epoch}.pth").unlink()
    print(f"source best_epoch={best_epoch + 1}", flush=True)

    module.model.cpu()
    module.model = module.optimizer = module.data = module.target_data = None
    sys.modules.pop("train_source", None)
    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_epoch + 1


def train_target(device, seed, epochs, target):
    seed_all(seed)
    module = import_stage(
        "train_target",
        [
            "--device", str(device),
            "--seed", str(seed),
            "--epochs", str(epochs),
        ],
    )
    captured = {}

    @torch.no_grad()
    def capture_evaluate(x, edge_index, edge_weight, _y, model):
        model.eval()
        embedding = model.feat_bottleneck(x, edge_index, edge_weight)
        logits = model.feat_classifier(embedding)
        select = all_valid_mask(target)
        mask = select.to(logits.device)
        labels = target.y[select].long().to(logits.device)
        loss = F.cross_entropy(logits[mask], labels)
        pred = logits[mask].argmax(1)
        captured.update(
            embedding=embedding[mask].cpu().numpy(),
            probability=F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy(),
            prediction=pred.cpu().numpy(),
        )
        return float((pred == labels).float().mean().item()), loss

    module.evaluate = capture_evaluate
    module.train_target(module.data, module.perturbed_edge_weight)
    if not captured:
        raise RuntimeError("GraphCTA target evaluation did not run")
    return captured


def evaluate(captured, target):
    mask = all_valid_mask(target)
    y = target.y[mask].cpu().numpy()
    sens = target.sens_labels[mask].cpu().numpy()
    pred = captured["prediction"]
    probability = captured["probability"]
    sens_ok = (sens == 0) | (sens == 1)
    dp, eo = fair_metric(pred[sens_ok], y[sens_ok], sens[sens_ok])
    metrics = {
        "ACC": float(accuracy_score(y, pred) * 100),
        "AUC": float(roc_auc_score(y, probability) * 100),
        "DP": float(dp * 100),
        "EO": float(eo * 100),
    }
    return metrics, captured["embedding"], y, sens


def run_one(
    dataset,
    run_idx,
    seed,
    gpu,
    source_epochs,
    target_epochs,
    patience,
    gpu_poll_seconds,
):
    seed_all(seed)
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
    if gpu >= 0:
        torch.cuda.set_device(gpu)

    source_id, target_id = PAIRS[dataset]
    print(
        f"dataset={dataset} run={run_idx + 1} seed={seed} "
        f"source={source_id} target={target_id} device={device}",
        flush=True,
    )
    source = get_dataset(SimpleNamespace(dataset=dataset), source_id)
    target = get_dataset(SimpleNamespace(dataset=dataset), target_id)
    if source.x.shape[1] != target.x.shape[1]:
        raise ValueError(f"{dataset}: source/target feature dimensions differ")
    install_pair_dataset(graphcta_view(source), graphcta_view(target, is_target=True))

    min_free_mib = DATASET_MIN_FREE_MIB[dataset]
    if dataset == "pokec":
        print(
            f"PYTORCH_CUDA_ALLOC_CONF="
            f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}",
            flush=True,
        )
    wait_for_dataset_gpu(
        dataset, gpu, min_free_mib, gpu_poll_seconds, stage="source"
    )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=f"{dataset}_run{run_idx + 1}_", dir=WORK_DIR) as work:
        os.chdir(work)
        try:
            best_epoch = train_source(device, seed, source_epochs, patience)
            wait_for_dataset_gpu(
                dataset, gpu, min_free_mib, gpu_poll_seconds, stage="target"
            )
            captured = train_target(device, seed, target_epochs, target)
        finally:
            os.chdir(previous_cwd)

    metrics, embedding, y, sens = evaluate(captured, target)
    print(f"target all(valid) metrics={json.dumps(metrics, sort_keys=True)}", flush=True)
    if run_idx == 0:
        save_visualization_embeddings(
            ROOT / "visualization" / "embeddings",
            "GraphCTA",
            dataset,
            embedding,
            y=y,
            sens=sens,
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output = RUN_DIR / f"GraphCTA_{dataset}_run{run_idx + 1}.json"
    output.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "run": run_idx + 1,
                "seed": seed,
                "source_best_epoch": best_epoch,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def aggregate():
    rows = []
    for dataset in PAIRS:
        runs = [
            json.loads(
                (RUN_DIR / f"GraphCTA_{dataset}_run{run_idx + 1}.json").read_text(
                    encoding="utf-8"
                )
            )
            for run_idx in range(len(SEEDS))
        ]
        stats = {}
        for metric in ("ACC", "AUC", "DP", "EO"):
            values = np.asarray([run["metrics"][metric] for run in runs])
            stats[metric] = (values.mean(), values.var(), values.std())
        rows.append((dataset, runs, stats))

    lines = [
        "# GraphCTA Results (5 runs)",
        "",
        "Source checkpoint selection uses source validation loss; target adaptation uses no target labels.",
        "Final evaluation is target `all(valid)`; Pokec `y=-1` is excluded.",
        "Variance is population variance (`ddof=0`); embeddings are exported from run 1 only.",
        "",
        "| Dataset | Metric | Mean | Variance | Std |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, _, stats in rows:
        for metric, (mean, variance, std) in stats.items():
            lines.append(
                f"| {dataset} | {metric} | {mean:.4f} | {variance:.6f} | {std:.4f} |"
            )

    lines += [
        "",
        "| Dataset | Run | Seed | Source Best Epoch | ACC | AUC | DP | EO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, runs, _ in rows:
        for run in runs:
            metrics = run["metrics"]
            lines.append(
                f"| {dataset} | {run['run']} | {run['seed']} | "
                f"{run['source_best_epoch']} | {metrics['ACC']:.4f} | "
                f"{metrics['AUC']:.4f} | {metrics['DP']:.4f} | {metrics['EO']:.4f} |"
            )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch(task, resources, args):
    gpu = resources.get()
    dataset, run_idx, seed = task
    log_path = LOG_DIR / f"{dataset}_run{run_idx + 1}.log"
    try:
        env = os.environ.copy()
        if dataset == "pokec" and not env.get("PYTORCH_CUDA_ALLOC_CONF"):
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        with log_path.open("w", encoding="utf-8") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--dataset", dataset,
                    "--run-idx", str(run_idx),
                    "--seed", str(seed),
                    "--gpu", str(gpu),
                    "--source-epochs", str(args.source_epochs),
                    "--target-epochs", str(args.target_epochs),
                    "--patience", str(args.patience),
                    "--gpu-poll-seconds", str(args.gpu_poll_seconds),
                ],
                cwd=HERE,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        resources.put(gpu)
    return dataset, run_idx


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphCTA for 5 seeds on bailA, germanA, pokec, and syn."
    )
    parser.add_argument("--source-epochs", type=int, default=SOURCE_EPOCHS)
    parser.add_argument("--target-epochs", type=int, default=TARGET_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--gpu-poll-seconds",
        type=int,
        default=GPU_POLL_SECONDS,
        help="seconds between dataset GPU-memory checks",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=PAIRS, default="bailA", help=argparse.SUPPRESS)
    parser.add_argument("--run-idx", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=SEEDS[0], help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if (
        args.source_epochs < 1
        or args.target_epochs < 3
        or args.patience < 1
        or any(value < 0 for value in DATASET_MIN_FREE_MIB.values())
        or args.gpu_poll_seconds < 1
    ):
        parser.error(
            "source_epochs >= 1, target_epochs >= 3, patience >= 1, "
            "all dataset GPU floors >= 0, and gpu_poll_seconds >= 1 are required"
        )
    if args.worker:
        run_one(
            args.dataset,
            args.run_idx,
            args.seed,
            args.gpu,
            args.source_epochs,
            args.target_epochs,
            args.patience,
            args.gpu_poll_seconds,
        )
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (dataset, run_idx, seed)
        for run_idx, seed in enumerate(SEEDS)
        for dataset in PAIRS
    ]
    devices = list(AVAILABLE_GPUS)
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not devices or len(devices) != len(set(devices)):
        parser.error("AVAILABLE_GPUS must contain unique GPU ids")
    invalid = [gpu for gpu in devices if gpu < 0 or gpu >= torch.cuda.device_count()]
    if invalid:
        parser.error(f"invalid GPU ids in AVAILABLE_GPUS: {invalid}")
    resources = Queue()
    for device in devices:
        resources.put(device)

    print(f"launching {len(tasks)} runs on devices={devices}", flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(launch, task, resources, args) for task in tasks]
        for future in as_completed(futures):
            dataset, run_idx = future.result()
            print(f"done: {dataset} run{run_idx + 1}", flush=True)
    aggregate()
    print(f"summary: {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
