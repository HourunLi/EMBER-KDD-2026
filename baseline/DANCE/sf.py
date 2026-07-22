# coding=utf-8
"""Source-only/source-free inference runner for the DANCE fairness baseline.

The original DANCE code trains on a source graph and evaluates the resulting
model on a shifted target graph.  This entry point keeps that minimal
source-only protocol: train on source labels, serialize the model, release the
source graph, load the target graph, and run one direct target forward pass.
There is deliberately no target optimizer, entropy minimization, pseudo-label,
or target-validation model selection.  Target annotations are read only by
the final all-valid evaluator/exporter.  Pokec nodes with y=-1 remain in the
target graph for message passing but are excluded from metrics and artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SFFGNN_DIR = REPO_ROOT / "SFFGNN"

# DANCE's modules use historical top-level imports (``from utils import *``).
# Resolve those imports before importing SFFGNN's package-qualified API.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import MLP_classifier, MLP_discriminator  # noqa: E402
from nets.gcn import StandGCN  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SFFGNN_DIR) not in sys.path:
    sys.path.append(str(SFFGNN_DIR))

from SFFGNN.dataset import get_dataset  # noqa: E402
from SFFGNN.utils import fair_metric  # noqa: E402
from visualization.export_utils import save_visualization_embeddings  # noqa: E402


METHOD_NAME = "DANCE-SF"
EMBEDDING_METHOD_NAME = "DANCE"
DEFAULT_DATASETS = ("bailA", "germanA", "pokec", "syn")
# Match the five-run convention used by the existing SFDA baseline launchers.
DEFAULT_SEEDS = (1111, 1112, 1113, 1114, 1115)
DATASET_PAIRS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

DEFAULT_RESULT_DIR = SCRIPT_DIR / "results" / "dance_sf"
SUMMARY_FILENAME = "DANCE-SF_summary.md"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
    return torch.device(device_name)


def load_graph(dataset: str, domain_id: str, device: torch.device):
    """Load exactly through SFFGNN/dataset.py, including feature scaling."""
    data = get_dataset(SimpleNamespace(dataset=dataset), domain_id)
    return data.to(device)


def valid_label_mask(data) -> torch.Tensor:
    return (data.y == 0) | (data.y == 1)


def source_train_mask(data) -> torch.Tensor:
    """Preserve DANCE's supervised source split and filter Pokec y=-1."""
    return data.train_mask.bool() & valid_label_mask(data)


def all_valid_mask(data) -> torch.Tensor:
    split_mask = data.train_mask | data.val_mask | data.test_mask
    return split_mask & valid_label_mask(data)


def build_modules(num_features: int, args, device: torch.device):
    """Instantiate the unchanged DANCE GCN, classifier and discriminator."""
    dance_args = SimpleNamespace(
        num_features=num_features,
        hidden=args.hidden,
        feature_dim=args.hidden,
        num_classes=1,
        n_layer=args.n_layer,
        dropout=args.dropout,
        activation=args.activation,
        device=device,
    )
    encoder = StandGCN(dance_args).to(device)
    classifier = MLP_classifier(dance_args).to(device)
    discriminator = MLP_discriminator(dance_args).to(device)
    return encoder, classifier, discriminator


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def checkpoint_path(result_dir: Path, dataset: str, run_idx: int, seed: int) -> Path:
    return result_dir / "checkpoints" / (
        f"dance_sf_{dataset}_run{run_idx}_seed{seed}_final.pt"
    )


def result_path(result_dir: Path, dataset: str, run_idx: int) -> Path:
    return result_dir / "runs" / f"dance_sf_{dataset}_run{run_idx}.json"


def summary_path(result_dir: Path) -> Path:
    return result_dir / SUMMARY_FILENAME


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def source_pretrain(
    source_data,
    dataset: str,
    run_idx: int,
    args,
    device: torch.device,
    result_dir: Path,
) -> Path:
    """Train DANCE on source only and persist transferable source knowledge."""
    source_mask = source_train_mask(source_data)
    if int(source_mask.sum().item()) == 0:
        raise ValueError(f"{dataset}: source graph has no valid binary labels")

    encoder, classifier, discriminator = build_modules(
        source_data.x.size(1), args, device
    )
    optimizer_encoder = torch.optim.Adam(
        encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_classifier = torch.optim.Adam(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.discriminator_lr,
        weight_decay=args.weight_decay,
    )

    class_target = source_data.y[source_mask].float().unsqueeze(1)
    sensitive_target = source_data.sens_labels.float().view(-1)

    for epoch in range(1, args.source_epochs + 1):
        # Original DANCE classification block (sigmoid classifier + MSE).
        set_requires_grad(discriminator, False)
        encoder.train()
        classifier.train()
        for _ in range(args.classifier_steps):
            optimizer_encoder.zero_grad()
            optimizer_classifier.zero_grad()
            source_features = encoder(source_data.x, source_data.edge_index)
            source_probability = classifier(source_features)
            classification_loss = F.mse_loss(
                source_probability[source_mask], class_target
            )
            classification_loss.backward()
            optimizer_encoder.step()
            optimizer_classifier.step()

        # Fit the source sensitive discriminator on detached source features.
        set_requires_grad(discriminator, True)
        encoder.eval()
        discriminator.train()
        for _ in range(args.discriminator_steps):
            optimizer_discriminator.zero_grad()
            with torch.no_grad():
                detached_features = encoder(source_data.x, source_data.edge_index)
            sensitive_probability = discriminator(detached_features).view(-1)
            discriminator_loss = F.mse_loss(
                sensitive_probability, sensitive_target
            )
            discriminator_loss.backward()
            optimizer_discriminator.step()

        # DANCE adversarial debiasing block: update only the encoder so that
        # sensitive membership is maximally ambiguous.
        set_requires_grad(discriminator, False)
        encoder.train()
        discriminator.eval()
        for _ in range(args.adversarial_steps):
            optimizer_encoder.zero_grad()
            source_features = encoder(source_data.x, source_data.edge_index)
            sensitive_probability = discriminator(source_features).view(-1)
            fairness_loss = F.mse_loss(
                sensitive_probability,
                torch.full_like(sensitive_probability, 0.5),
            )
            (args.source_fair_weight * fairness_loss).backward()
            optimizer_encoder.step()

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.source_epochs:
            print(
                f"[{dataset} run{run_idx}] source epoch={epoch}/{args.source_epochs} "
                f"L_cls={classification_loss.item():.6f} "
                f"L_disc={discriminator_loss.item():.6f} "
                f"L_fair={fairness_loss.item():.6f}"
            )

    checkpoint = {
        "method": METHOD_NAME,
        "dataset": dataset,
        "run_idx": run_idx,
        "seed": args.seed,
        "num_features": int(source_data.x.size(1)),
        "hidden": int(args.hidden),
        "encoder_state": {
            key: value.detach().cpu() for key, value in encoder.state_dict().items()
        },
        "classifier_state": {
            key: value.detach().cpu() for key, value in classifier.state_dict().items()
        },
        "discriminator_state": {
            key: value.detach().cpu()
            for key, value in discriminator.state_dict().items()
        },
    }
    path = checkpoint_path(result_dir, dataset, run_idx, args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"[{dataset} run{run_idx}] source checkpoint saved: {path}")
    return path


def load_source_modules(checkpoint: dict, args, device: torch.device):
    encoder, classifier, discriminator = build_modules(
        int(checkpoint["num_features"]), args, device
    )
    encoder.load_state_dict(checkpoint["encoder_state"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    discriminator.load_state_dict(checkpoint["discriminator_state"])
    return encoder, classifier, discriminator


def load_target_model(
    target_num_features: int,
    checkpoint_file: Path,
    dataset: str,
    args,
    device: torch.device,
):
    """Load the source-trained model for direct target inference only."""
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if int(checkpoint["num_features"]) != int(target_num_features):
        raise ValueError(
            f"{dataset}: source/target feature dimensions differ: "
            f"{checkpoint['num_features']} vs {target_num_features}"
        )

    encoder, classifier, discriminator = load_source_modules(
        checkpoint, args, device
    )
    encoder.eval()
    classifier.eval()
    discriminator.eval()
    return encoder, classifier


def evaluate_target(encoder, classifier, target_data) -> dict[str, float]:
    """Final-only target evaluation using SFFGNN's all-valid convention."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    mask = all_valid_mask(target_data)
    if int(mask.sum().item()) == 0:
        raise ValueError("Target graph has no valid 0/1 labels in all split")

    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        features = encoder(target_data.x, target_data.edge_index)
        probability = classifier(features).view(-1)

    mask_np = mask.detach().cpu().numpy()
    y_true = target_data.y.detach().cpu().numpy()[mask_np].astype(int)
    sensitive = (
        target_data.sens_labels.detach().cpu().numpy()[mask_np].astype(int)
    )
    probability_np = probability.detach().cpu().numpy()[mask_np]
    prediction = (probability_np > 0.5).astype(int)

    accuracy = accuracy_score(y_true, prediction) * 100.0
    auc = (
        roc_auc_score(y_true, probability_np) * 100.0
        if len(np.unique(y_true)) == 2
        else float("nan")
    )
    dp, eo = fair_metric(prediction, y_true, sensitive)
    return {
        "acc": float(accuracy),
        "auc": float(auc),
        "dp": float(dp * 100.0),
        "eo": float(eo * 100.0),
        "num_eval_nodes": int(mask.sum().item()),
        "num_unlabeled_filtered": int((target_data.y == -1).sum().item()),
    }


def export_target_embeddings(
    encoder,
    target_data,
    dataset: str,
) -> tuple[Path, Path]:
    mask = all_valid_mask(target_data)
    encoder.eval()
    with torch.no_grad():
        features = encoder(target_data.x, target_data.edge_index)
    return save_visualization_embeddings(
        REPO_ROOT / "visualization" / "embeddings",
        EMBEDDING_METHOD_NAME,
        dataset,
        features[mask].detach().cpu().numpy(),
        y=target_data.y[mask].detach().cpu().numpy(),
        sens=target_data.sens_labels[mask].detach().cpu().numpy(),
    )


def run_worker(args) -> dict:
    if args.dataset not in DATASET_PAIRS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if args.seed is None:
        raise ValueError("worker mode requires --seed")
    if args.source_epochs <= 0:
        raise ValueError("--source_epochs must be positive")
    if min(
        args.classifier_steps,
        args.discriminator_steps,
        args.adversarial_steps,
    ) <= 0:
        raise ValueError("all alternating DANCE step counts must be positive")
    if args.hidden <= 0 or args.n_layer <= 0:
        raise ValueError("--hidden and --n_layer must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    result_dir = Path(args.result_dir).resolve()
    source_id = args.source_id or DATASET_PAIRS[args.dataset][0]
    target_id = args.target_id or DATASET_PAIRS[args.dataset][1]
    print(
        f"[{METHOD_NAME}] dataset={args.dataset} run={args.run_idx} seed={args.seed} "
        f"source={source_id} target={target_id} device={device}"
    )

    source_data = load_graph(args.dataset, source_id, device)
    checkpoint_file = source_pretrain(
        source_data,
        args.dataset,
        args.run_idx,
        args,
        device,
        result_dir,
    )

    # The source graph and source-side modules are out of scope from here on.
    del source_data
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_data = load_graph(args.dataset, target_id, device)
    # Load source weights and perform one direct target forward pass.  No
    # target optimizer or target-side adaptation is performed.
    encoder, classifier = load_target_model(
        target_data.x.size(1),
        checkpoint_file,
        args.dataset,
        args,
        device,
    )

    if (
        args.save_visualization_embeddings
        and args.run_idx == args.visualization_run_idx
    ):
        feat_path, labels_path = export_target_embeddings(
            encoder, target_data, args.dataset
        )
        print(f"[{METHOD_NAME}] embeddings saved: {feat_path}")
        print(f"[{METHOD_NAME}] labels saved: {labels_path}")

    metrics = evaluate_target(encoder, classifier, target_data)
    result = {
        "method": METHOD_NAME,
        "dataset": args.dataset,
        "source_id": source_id,
        "target_id": target_id,
        "run_idx": args.run_idx,
        "seed": args.seed,
        "evaluation_protocol": "source_only_direct_target_inference",
        "evaluation_split": "all_valid",
        "metrics": metrics,
        "hyperparameters": {
            "source_epochs": args.source_epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
            "source_fair_weight": args.source_fair_weight,
        },
    }
    output_file = result_path(result_dir, args.dataset, args.run_idx)
    atomic_write_json(output_file, result)
    print(f"[{METHOD_NAME}] result saved: {output_file}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return result


def parse_seed_list(seed_specification: str, runs: int) -> list[int]:
    seeds = [
        int(item.strip())
        for item in seed_specification.split(",")
        if item.strip()
    ]
    if len(seeds) < runs:
        raise ValueError(
            f"--seeds provides {len(seeds)} values, but --runs={runs}"
        )
    return seeds[:runs]


def parse_gpu_list(gpu_specification: str) -> list[str]:
    devices = [
        item.strip() for item in gpu_specification.split(",") if item.strip()
    ]
    if not devices:
        raise ValueError("--gpus must contain at least one GPU id or 'cpu'")
    if not torch.cuda.is_available():
        # Preserve process-level parallelism on CPU-only hosts without
        # accidentally launching an excessive number of memory-heavy graphs.
        cpu_slots = min(len(devices), max(1, min(4, os.cpu_count() or 1)))
        return ["cpu"] * cpu_slots
    return devices


def finite_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "var": float(array.var(ddof=0)),
    }


def load_dataset_results(
    result_dir: Path, dataset: str, runs: int
) -> list[dict] | None:
    rows = []
    for run_idx in range(runs):
        path = result_path(result_dir, dataset, run_idx)
        if not path.exists():
            return None
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def aggregate_results(args) -> tuple[Path, list[str]]:
    result_dir = Path(args.result_dir).resolve()
    metrics = ("acc", "auc", "dp", "eo")
    completed = []
    dataset_rows = {}
    dataset_stats = {}

    for dataset in args.datasets:
        rows = load_dataset_results(result_dir, dataset, args.runs)
        if rows is None:
            continue
        completed.append(dataset)
        dataset_rows[dataset] = rows
        dataset_stats[dataset] = {
            metric: finite_stats(
                [float(row["metrics"][metric]) for row in rows]
            )
            for metric in metrics
        }

    lines = [
        f"# {METHOD_NAME}: {args.runs}-run summary",
        "",
        "All metrics are percentages on the union of train/validation/test masks; "
        "Pokec y=-1 nodes are excluded. Population variance/std (ddof=0) are reported.",
        "",
        "## Per-run results",
        "",
        "|Dataset|Run|Seed|ACC|AUC|DP|EO|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in args.datasets:
        for row in dataset_rows.get(dataset, []):
            value = row["metrics"]
            lines.append(
                f"|{dataset}|{row['run_idx']}|{row['seed']}|"
                f"{value['acc']:.4f}|{value['auc']:.4f}|"
                f"{value['dp']:.4f}|{value['eo']:.4f}|"
            )

    lines.extend(
        [
            "",
            "## Mean ± standard deviation",
            "",
            "|Dataset|ACC|AUC|DP|EO|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in args.datasets:
        if dataset not in dataset_stats:
            continue
        stats = dataset_stats[dataset]
        lines.append(
            f"|{dataset}|"
            + "|".join(
                f"{stats[metric]['mean']:.2f} ± {stats[metric]['std']:.2f}"
                for metric in metrics
            )
            + "|"
        )

    lines.extend(
        [
            "",
            "## Mean and variance",
            "",
            "|Dataset|ACC mean|ACC var|AUC mean|AUC var|DP mean|DP var|EO mean|EO var|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in args.datasets:
        if dataset not in dataset_stats:
            continue
        stats = dataset_stats[dataset]
        lines.append(
            f"|{dataset}|"
            f"{stats['acc']['mean']:.6f}|{stats['acc']['var']:.6f}|"
            f"{stats['auc']['mean']:.6f}|{stats['auc']['var']:.6f}|"
            f"{stats['dp']['mean']:.6f}|{stats['dp']['var']:.6f}|"
            f"{stats['eo']['mean']:.6f}|{stats['eo']['var']:.6f}|"
        )

    output = summary_path(result_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output, completed


def worker_command(args, dataset: str, run_idx: int, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
        "--dataset",
        dataset,
        "--run_idx",
        str(run_idx),
        "--seed",
        str(seed),
        "--result_dir",
        str(Path(args.result_dir).resolve()),
        "--source_epochs",
        str(args.source_epochs),
        "--classifier_steps",
        str(args.classifier_steps),
        "--discriminator_steps",
        str(args.discriminator_steps),
        "--adversarial_steps",
        str(args.adversarial_steps),
        "--lr",
        str(args.lr),
        "--discriminator_lr",
        str(args.discriminator_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--hidden",
        str(args.hidden),
        "--n_layer",
        str(args.n_layer),
        "--dropout",
        str(args.dropout),
        "--activation",
        args.activation,
        "--source_fair_weight",
        str(args.source_fair_weight),
        "--print_every",
        str(args.print_every),
        "--visualization_run_idx",
        str(args.visualization_run_idx),
    ]
    if args.save_visualization_embeddings:
        command.append("--save_visualization_embeddings")
    else:
        command.append("--no_save_visualization_embeddings")
    return command


def launch_parallel(args) -> None:
    for dataset in args.datasets:
        if dataset not in DATASET_PAIRS:
            raise ValueError(f"Unsupported dataset: {dataset}")
    if not 0 <= args.visualization_run_idx < args.runs:
        raise ValueError("--visualization_run_idx must be a valid run index")

    seeds = parse_seed_list(args.seeds, args.runs)
    gpu_ids = parse_gpu_list(args.gpus)
    result_dir = Path(args.result_dir).resolve()
    log_dir = result_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {"dataset": dataset, "run_idx": run_idx, "seed": seeds[run_idx]}
        for dataset in args.datasets
        for run_idx in range(args.runs)
    ]
    # Fixed result names are overwritten atomically.  Remove only the exact
    # requested run files so failed new jobs cannot be mistaken for old runs.
    for job in jobs:
        old_result = result_path(result_dir, job["dataset"], job["run_idx"])
        if old_result.exists():
            old_result.unlink()

    pending = list(jobs)
    available_devices = list(gpu_ids)
    active = []
    failures = []
    print(
        f"[Launcher] jobs={len(jobs)} devices={gpu_ids} result_dir={result_dir}"
    )

    while pending or active:
        while pending and available_devices:
            job = pending.pop(0)
            gpu_id = available_devices.pop(0)
            log_file_path = log_dir / (
                f"dance_sf_{job['dataset']}_run{job['run_idx']}_seed{job['seed']}.log"
            )
            command = worker_command(
                args, job["dataset"], job["run_idx"], job["seed"]
            )
            environment = os.environ.copy()
            if gpu_id.lower() == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = ""
                command.extend(["--device", "cpu"])
            else:
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
                command.extend(["--device", "cuda:0"])

            log_handle = open(log_file_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(SCRIPT_DIR),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append(
                {
                    "process": process,
                    "gpu_id": gpu_id,
                    "job": job,
                    "log_handle": log_handle,
                    "log_path": log_file_path,
                }
            )
            print(
                f"[Launcher] start {job['dataset']} run{job['run_idx']} "
                f"seed={job['seed']} device={gpu_id} log={log_file_path}"
            )

        still_active = []
        for item in active:
            return_code = item["process"].poll()
            if return_code is None:
                still_active.append(item)
                continue
            item["log_handle"].close()
            available_devices.append(item["gpu_id"])
            job = item["job"]
            if return_code != 0:
                failures.append((job, item["log_path"], return_code))
                print(
                    f"[Launcher] FAILED {job['dataset']} run{job['run_idx']} "
                    f"return_code={return_code} log={item['log_path']}"
                )
            else:
                print(f"[Launcher] done {job['dataset']} run{job['run_idx']}")
        active = still_active
        if pending or active:
            time.sleep(args.poll_seconds)

    output, completed = aggregate_results(args)
    print(f"[Launcher] summary saved: {output}")
    print(f"[Launcher] completed datasets: {completed}")
    if failures:
        for job, log_file_path, return_code in failures:
            print(
                f"  failed dataset={job['dataset']} run={job['run_idx']} "
                f"return_code={return_code} log={log_file_path}"
            )
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DANCE source-only direct target evaluation on SFFGNN datasets"
    )
    parser.add_argument(
        "--mode", choices=("launch", "worker", "aggregate"), default="launch"
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument("--result_dir", type=str, default=str(DEFAULT_RESULT_DIR))

    # Single-run worker arguments.
    parser.add_argument("--dataset", choices=list(DATASET_PAIRS), default=None)
    parser.add_argument("--source_id", type=str, default=None)
    parser.add_argument("--target_id", type=str, default=None)
    parser.add_argument("--run_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    # DANCE architecture/training defaults.  The original repository uses 500
    # source epochs; one step per block keeps the three alternating objectives
    # while avoiding its dataset-specific 5-15x nested update multipliers.
    parser.add_argument("--source_epochs", type=int, default=500)
    parser.add_argument("--classifier_steps", type=int, default=1)
    parser.add_argument("--discriminator_steps", type=int, default=1)
    parser.add_argument("--adversarial_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discriminator_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--activation", choices=("ident", "sigmoid", "LeakyReLU"), default="LeakyReLU"
    )
    parser.add_argument("--source_fair_weight", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=50)

    parser.add_argument(
        "--save_visualization_embeddings", action="store_true", default=True
    )
    parser.add_argument(
        "--no_save_visualization_embeddings",
        dest="save_visualization_embeddings",
        action="store_false",
    )
    parser.add_argument("--visualization_run_idx", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        if args.dataset is None:
            raise ValueError("worker mode requires --dataset")
        run_worker(args)
    elif args.mode == "aggregate":
        output, completed = aggregate_results(args)
        print(f"summary saved: {output}")
        print(f"completed datasets: {completed}")
    else:
        launch_parallel(args)


if __name__ == "__main__":
    main()
