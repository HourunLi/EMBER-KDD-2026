# coding=utf-8
"""
GRADE-SF runner for the SFFGNN fair domain adaptation datasets.

This file keeps the original GRADE-N spirit: a shared GCN encoder, a linear
classifier, and graph subtree discrepancy over layer-wise representations.
The source-free change is restricted to the adaptation signal. Source graph
representations are compressed into layer-wise class-conditional moments after
source pretraining; target adaptation only reads those moments and the source
checkpoint, never the source graph or source node samples.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError:
    accuracy_score = None
    roc_auc_score = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SFFGNN_DIR = REPO_ROOT / "SFFGNN"

# SFFGNN/dataset.py imports "utils" as a top-level module. Put SFFGNN_DIR
# before this script directory so it resolves to SFFGNN/utils.py, not
# baseline/GRADE/utils.py.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SFFGNN_DIR) not in sys.path:
    sys.path.insert(0, str(SFFGNN_DIR))

from SFFGNN.dataset import get_dataset
from SFFGNN.utils import fair_metric


METHOD_NAME = "GRADE-SF"
DATASET_ID_MAP = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

RESULT_DIR = SCRIPT_DIR / "results" / "grade_sf"
RUN_DIR = RESULT_DIR / "runs"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
LOG_DIR = RESULT_DIR / "logs"
SUMMARY_FILE = RESULT_DIR / "GRADE-SF_summary.md"


def seed_everything(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cuda_id):
    if cuda_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(cuda_id)
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def load_graph(dataset, domain_id, device):
    data = get_dataset(SimpleNamespace(dataset=dataset), domain_id)
    return data.to(device)


def valid_label_mask(data):
    # Pokec contains unlabeled nodes with y = -1. Match SFFGNN: exclude them
    # from supervised source training, source statistics, and final metrics.
    return data.y >= 0


def source_train_mask(data):
    return data.train_mask & valid_label_mask(data)


def eval_mask(data, split):
    if split == "train":
        mask = data.train_mask
    elif split == "val":
        mask = data.val_mask
    elif split == "test":
        mask = data.test_mask
    elif split == "all":
        mask = data.train_mask | data.val_mask | data.test_mask
    else:
        raise ValueError(f"Unknown eval split: {split}")
    return mask & valid_label_mask(data)


def infer_num_classes(data):
    labels = data.y[valid_label_mask(data)]
    if labels.numel() == 0:
        raise ValueError("No valid labels found in dataset")
    return max(2, int(labels.max().item()) + 1)


class GRADEBackbone(nn.Module):
    """PyG implementation of the original GRADE-N GCN + linear classifier."""

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_feats, n_hidden))
        for _ in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden))
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, return_reps=False):
        reps = []
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x, edge_index)
            x = F.relu(x)
            reps.append(x)
        logits = self.fc(x)
        reps.append(logits)
        if return_reps:
            return logits, reps
        return logits


def build_model(data, args, device, num_classes=None):
    if num_classes is None:
        num_classes = infer_num_classes(data)
    return GRADEBackbone(
        in_feats=data.x.size(1),
        n_hidden=args.n_hidden,
        n_classes=num_classes,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)


@torch.no_grad()
def compute_source_stats(model, data, mask, num_classes, eps):
    model.eval()
    _, reps = model(data.x, data.edge_index, return_reps=True)
    labels = data.y.long()
    stats = []

    for rep in reps:
        means = []
        variances = []
        counts = []
        for class_id in range(num_classes):
            class_mask = mask & (labels == class_id)
            values = rep[class_mask]
            counts.append(int(values.size(0)))
            if values.size(0) == 0:
                means.append(torch.zeros(rep.size(1), device=rep.device))
                variances.append(torch.ones(rep.size(1), device=rep.device))
                continue

            means.append(values.mean(dim=0))
            variances.append(values.var(dim=0, unbiased=False).clamp_min(eps))

        stats.append({
            "mean": torch.stack(means, dim=0).detach().cpu(),
            "var": torch.stack(variances, dim=0).detach().cpu(),
            "count": torch.tensor(counts, dtype=torch.long),
        })

    return stats


def move_stats_to_device(stats, device):
    moved = []
    for layer_stats in stats:
        moved.append({
            "mean": layer_stats["mean"].to(device),
            "var": layer_stats["var"].to(device),
            "count": layer_stats["count"].to(device),
        })
    return moved


def build_target_weights(probs, confidence_ratio):
    weights = probs.detach()
    if confidence_ratio >= 1.0:
        return weights

    confidence = weights.max(dim=1).values
    keep_ratio = min(max(confidence_ratio, 1e-6), 1.0)
    threshold = torch.quantile(confidence, 1.0 - keep_ratio)
    keep = confidence >= threshold
    return weights * keep.float().unsqueeze(1)


def source_free_gsd_loss(reps, probs, source_stats, args):
    weights = build_target_weights(probs, args.confidence_ratio)
    total_weight = weights.sum().clamp_min(args.stat_eps)
    total_loss = reps[0].new_tensor(0.0)
    valid_terms = 0

    for rep, layer_stats in zip(reps, source_stats):
        layer_loss = rep.new_tensor(0.0)
        layer_terms = 0
        for class_id in range(weights.size(1)):
            if int(layer_stats["count"][class_id].item()) == 0:
                continue

            class_weight = weights[:, class_id]
            mass = class_weight.sum()
            if mass.item() <= args.stat_eps:
                continue

            src_mean = layer_stats["mean"][class_id]
            src_var = layer_stats["var"][class_id].clamp_min(args.stat_eps)

            weighted_rep = rep * class_weight.unsqueeze(1)
            tgt_mean = weighted_rep.sum(dim=0) / mass.clamp_min(args.stat_eps)
            centered = rep - tgt_mean.unsqueeze(0)
            tgt_var = (
                class_weight.unsqueeze(1) * centered.pow(2)
            ).sum(dim=0) / mass.clamp_min(args.stat_eps)
            tgt_var = tgt_var.clamp_min(args.stat_eps)

            mean_loss = ((tgt_mean - src_mean).pow(2) / src_var).mean()
            var_loss = F.smooth_l1_loss(
                torch.log(tgt_var),
                torch.log(src_var),
                reduction="mean",
            )
            class_prior = mass / total_weight
            layer_loss = layer_loss + class_prior * (mean_loss + args.var_weight * var_loss)
            layer_terms += 1

        if layer_terms > 0:
            total_loss = total_loss + layer_loss
            valid_terms += 1

    if valid_terms == 0:
        return reps[0].new_tensor(0.0)
    return total_loss / valid_terms


def l2_sp_loss(model, source_state):
    loss = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name not in source_state:
            continue
        current = (param - source_state[name]).pow(2).mean()
        loss = current if loss is None else loss + current
    if loss is None:
        return next(model.parameters()).new_tensor(0.0)
    return loss


def checkpoint_path(dataset, run_idx):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{METHOD_NAME}_{dataset}_run{run_idx}_final.pt"


def run_result_path(dataset, run_idx):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{dataset}_run{run_idx}.json"


def dataset_result_path(dataset):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{dataset}_{METHOD_NAME}_results.txt"


def save_checkpoint(path, model, source_stats, dataset, source_id, run_idx, seed, args):
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "source_stats": source_stats,
            "dataset": dataset,
            "source_id": source_id,
            "run_idx": run_idx,
            "seed": seed,
            "model_args": {
                "num_classes": model.fc.out_features,
                "n_hidden": args.n_hidden,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            },
            "train_args": {
                "source_epochs": args.source_epochs,
                "target_epochs": args.target_epochs,
                "source_lr": args.source_lr,
                "target_lr": args.target_lr,
                "lambda_gsd": args.lambda_gsd,
                "lambda_sp": args.lambda_sp,
                "var_weight": args.var_weight,
                "confidence_ratio": args.confidence_ratio,
            },
        },
        path,
    )


def stage1_pretrain_source(dataset, source_id, run_idx, args, device):
    source_data = load_graph(dataset, source_id, device)
    model = build_model(source_data, args, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.source_lr,
        weight_decay=args.weight_decay,
    )
    train_mask = source_train_mask(source_data)
    if int(train_mask.sum().item()) == 0:
        raise ValueError(f"{dataset} source train split has no valid 0/1 labels")

    for epoch in range(1, args.source_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(source_data.x, source_data.edge_index)
        loss = F.cross_entropy(logits[train_mask], source_data.y[train_mask].long())
        loss.backward()
        optimizer.step()

        if args.verbose and (epoch % args.print_every == 0 or epoch == args.source_epochs):
            print(
                f"[{dataset} run{run_idx}] stage1 epoch={epoch} "
                f"L_source={loss.item():.6f}"
            )

    num_classes = model.fc.out_features
    source_stats = compute_source_stats(
        model,
        source_data,
        train_mask,
        num_classes,
        args.stat_eps,
    )
    ckpt_path = checkpoint_path(dataset, run_idx)
    save_checkpoint(
        ckpt_path,
        model,
        source_stats,
        dataset,
        source_id,
        run_idx,
        args.seed + run_idx,
        args,
    )

    del source_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path


def load_checkpoint(path, target_data, args, device):
    artifact = torch.load(path, map_location=device)
    num_classes = int(artifact["model_args"]["num_classes"])
    model = build_model(target_data, args, device, num_classes=num_classes)
    model.load_state_dict({k: v.to(device) for k, v in artifact["model_state"].items()})
    source_stats = move_stats_to_device(artifact["source_stats"], device)
    source_state = {k: v.to(device).detach().clone() for k, v in artifact["model_state"].items()}
    return model, source_stats, source_state


def stage2_adapt_target(dataset, target_id, run_idx, ckpt_path, args, device):
    target_data = load_graph(dataset, target_id, device)
    model, source_stats, source_state = load_checkpoint(
        ckpt_path,
        target_data,
        args,
        device,
    )

    before_metrics = evaluate_target(model, target_data, args.eval_split)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.target_lr)

    for epoch in range(1, args.target_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, reps = model(target_data.x, target_data.edge_index, return_reps=True)
        probs = F.softmax(logits, dim=1).clamp(min=1e-8)

        gsd_loss = source_free_gsd_loss(reps, probs, source_stats, args)
        loss = args.lambda_gsd * gsd_loss
        sp_value = None
        if args.lambda_sp > 0:
            sp_loss = l2_sp_loss(model, source_state)
            loss = loss + args.lambda_sp * sp_loss
            sp_value = sp_loss.item()
        loss.backward()
        optimizer.step()

        if args.verbose and (epoch % args.print_every == 0 or epoch == args.target_epochs):
            msg = (
                f"[{dataset} run{run_idx}] stage2 epoch={epoch} "
                f"L_gsd={gsd_loss.item():.6f}"
            )
            if sp_value is not None:
                msg += f" L_sp={sp_value:.6f}"
            print(msg)

    after_metrics = evaluate_target(model, target_data, args.eval_split)
    return before_metrics, after_metrics


@torch.no_grad()
def evaluate_target(model, target_data, split):
    model.eval()
    mask = eval_mask(target_data, split)
    if int(mask.sum().item()) == 0:
        raise ValueError("Target eval split has no valid labels")

    logits = model(target_data.x, target_data.edge_index)
    probs_all = F.softmax(logits, dim=1)
    pred = probs_all.argmax(dim=1)[mask].detach().cpu().numpy()
    labels = target_data.y[mask].detach().cpu().numpy()
    sens = target_data.sens_labels[mask].detach().cpu().numpy().astype(int)

    if probs_all.size(1) > 1:
        prob = probs_all[:, 1][mask].detach().cpu().numpy()
    else:
        prob = probs_all[:, 0][mask].detach().cpu().numpy()

    if accuracy_score is not None:
        acc = accuracy_score(labels, pred) * 100.0
    else:
        acc = float((pred == labels).mean() * 100.0)

    if roc_auc_score is not None and len(set(labels.tolist())) == 2 and probs_all.size(1) > 1:
        auc = roc_auc_score(labels, prob) * 100.0
    else:
        auc = float("nan")

    dp, eo = fair_metric(pred, labels, sens)
    return {"Acc": acc, "AUC": auc, "DP": dp * 100.0, "EO": eo * 100.0}


def write_run_result(dataset, source_id, target_id, run_idx, before_metrics, after_metrics, args):
    path = run_result_path(dataset, run_idx)
    payload = {
        "method": METHOD_NAME,
        "dataset": dataset,
        "source_id": source_id,
        "target_id": target_id,
        "run_idx": run_idx,
        "seed": args.seed + run_idx,
        "eval_split": args.eval_split,
        "target_before": before_metrics,
        "metrics": after_metrics,
        "hyperparameters": {
            "source_epochs": args.source_epochs,
            "target_epochs": args.target_epochs,
            "source_lr": args.source_lr,
            "target_lr": args.target_lr,
            "n_hidden": args.n_hidden,
            "n_layers": args.n_layers,
            "lambda_gsd": args.lambda_gsd,
            "lambda_sp": args.lambda_sp,
            "var_weight": args.var_weight,
            "confidence_ratio": args.confidence_ratio,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def run_single(args):
    source_id, target_id = DATASET_ID_MAP[args.dataset]
    seed_everything(args.seed + args.run_idx)
    device = get_device(args.cuda)

    print(
        f"[{METHOD_NAME}] dataset={args.dataset} run={args.run_idx} "
        f"source={source_id} target={target_id} device={device}"
    )

    ckpt_path = stage1_pretrain_source(args.dataset, source_id, args.run_idx, args, device)
    before_metrics, after_metrics = stage2_adapt_target(
        args.dataset,
        target_id,
        args.run_idx,
        ckpt_path,
        args,
        device,
    )
    out_path = write_run_result(
        args.dataset,
        source_id,
        target_id,
        args.run_idx,
        before_metrics,
        after_metrics,
        args,
    )
    print(f"[{METHOD_NAME}] run result saved: {out_path}")
    print(json.dumps({"before": before_metrics, "after": after_metrics}, indent=2))


def fmt_mean_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return "nan+/-nan"
    return f"{np.nanmean(arr):.2f}+/-{np.nanstd(arr):.2f}"


def aggregate_dataset(dataset, runs):
    rows = []
    for run_idx in range(runs):
        path = run_result_path(dataset, run_idx)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f)["metrics"])

    summary = {
        metric: fmt_mean_std([row[metric] for row in rows])
        for metric in ["Acc", "AUC", "DP", "EO"]
    }

    out = dataset_result_path(dataset)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Method: {METHOD_NAME}\n")
        f.write(f"Eval split: final {runs} runs, target labels used only for metrics\n")
        f.write("|Dataset|ACC|AUC|DP|EO|\n")
        f.write("|---|---:|---:|---:|---:|\n")
        f.write(
            f"|{dataset}|{summary['Acc']}|{summary['AUC']}|"
            f"{summary['DP']}|{summary['EO']}|\n"
        )
        f.write("\nRaw runs:\n")
        for idx, row in enumerate(rows):
            f.write(
                f"run{idx}: Acc={row['Acc']:.4f}, AUC={row['AUC']:.4f}, "
                f"DP={row['DP']:.4f}, EO={row['EO']:.4f}\n"
            )
    return summary


def aggregate_all(datasets, runs):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for dataset in datasets:
        summary = aggregate_dataset(dataset, runs)
        if summary is not None:
            summaries[dataset] = summary

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("|Dataset|ACC|AUC|DP|EO|\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for dataset in datasets:
            if dataset not in summaries:
                continue
            row = summaries[dataset]
            f.write(f"|{dataset}|{row['Acc']}|{row['AUC']}|{row['DP']}|{row['EO']}|\n")
    return SUMMARY_FILE


def parse_gpu_ids(gpus):
    ids = []
    for item in gpus.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "cpu":
            ids.append(-1)
        else:
            ids.append(int(item))
    return ids or [-1]


def launch_parallel(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    tasks = [(dataset, run_idx) for dataset in datasets for run_idx in range(args.runs)]
    gpu_ids = parse_gpu_ids(args.gpus)
    if not torch.cuda.is_available():
        gpu_ids = [-1]

    running = {}
    completed = set()
    task_queue = list(tasks)
    free_gpus = list(gpu_ids)

    def start_task(dataset, run_idx, gpu_id):
        log_path = LOG_DIR / f"{dataset}_run{run_idx}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--dataset",
            dataset,
            "--run_idx",
            str(run_idx),
            "--runs",
            str(args.runs),
            "--cuda",
            str(gpu_id),
            "--seed",
            str(args.seed),
            "--source_epochs",
            str(args.source_epochs),
            "--target_epochs",
            str(args.target_epochs),
            "--source_lr",
            str(args.source_lr),
            "--target_lr",
            str(args.target_lr),
            "--weight_decay",
            str(args.weight_decay),
            "--n_hidden",
            str(args.n_hidden),
            "--n_layers",
            str(args.n_layers),
            "--dropout",
            str(args.dropout),
            "--lambda_gsd",
            str(args.lambda_gsd),
            "--lambda_sp",
            str(args.lambda_sp),
            "--var_weight",
            str(args.var_weight),
            "--confidence_ratio",
            str(args.confidence_ratio),
            "--stat_eps",
            str(args.stat_eps),
            "--eval_split",
            args.eval_split,
            "--print_every",
            str(args.print_every),
        ]
        if args.verbose:
            cmd.append("--verbose")

        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(SCRIPT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        running[proc] = {
            "dataset": dataset,
            "run_idx": run_idx,
            "gpu": gpu_id,
            "log": log_file,
            "log_path": log_path,
        }
        print(f"[Launch] {dataset} run{run_idx} -> cuda={gpu_id}, log={log_path}")

    while task_queue or running:
        while task_queue and free_gpus:
            dataset, run_idx = task_queue.pop(0)
            start_task(dataset, run_idx, free_gpus.pop(0))

        time.sleep(2)
        for proc in list(running.keys()):
            ret = proc.poll()
            if ret is None:
                continue
            info = running.pop(proc)
            info["log"].close()
            free_gpus.append(info["gpu"])
            if ret != 0:
                raise RuntimeError(
                    f"{info['dataset']} run{info['run_idx']} failed, "
                    f"see {info['log_path']}"
                )

            completed.add((info["dataset"], info["run_idx"]))
            print(f"[Done] {info['dataset']} run{info['run_idx']}")
            dataset_runs = {(info["dataset"], idx) for idx in range(args.runs)}
            if dataset_runs.issubset(completed):
                summary = aggregate_dataset(info["dataset"], args.runs)
                print(f"[Dataset summary] {info['dataset']}: {summary}")

    out = aggregate_all(datasets, args.runs)
    print(f"[All done] summary saved: {out}")


def get_args():
    parser = argparse.ArgumentParser(description="GRADE-SF source-free runner")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dataset", choices=list(DATASET_ID_MAP.keys()), default="syn")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bailA", "germanA", "pokec", "syn"],
        choices=list(DATASET_ID_MAP.keys()),
    )
    parser.add_argument("--run_idx", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1111)

    parser.add_argument("--source_epochs", type=int, default=200)
    parser.add_argument("--target_epochs", type=int, default=100)
    parser.add_argument("--source_lr", type=float, default=0.001)
    parser.add_argument("--target_lr", type=float, default=0.0003)
    parser.add_argument("--lr", type=float, default=None, help="compatibility: overrides source_lr")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--lambda_gsd", type=float, default=0.01)
    parser.add_argument(
        "--lambda_entropy",
        type=float,
        default=0.0,
        help="deprecated compatibility flag; entropy loss is disabled in GRADE-SF",
    )
    parser.add_argument(
        "--lambda_prior",
        type=float,
        default=0.0,
        help="deprecated compatibility flag; source-prior loss is disabled in GRADE-SF",
    )
    parser.add_argument("--lambda_sp", type=float, default=0)
    parser.add_argument("--var_weight", type=float, default=0.1)
    parser.add_argument(
        "--confidence_ratio",
        type=float,
        default=1.0,
        help="fraction of most confident target nodes used for target moments",
    )
    parser.add_argument("--stat_eps", type=float, default=1e-6)

    parser.add_argument("--eval_split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = get_args()
    if cli_args.lr is not None:
        cli_args.source_lr = cli_args.lr

    if cli_args.aggregate_only:
        out_file = aggregate_all(cli_args.datasets, cli_args.runs)
        print(f"[Aggregate] summary saved: {out_file}")
    elif cli_args.worker:
        run_single(cli_args)
    else:
        launch_parallel(cli_args)
