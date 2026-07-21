# -*- coding: utf-8 -*-
"""GraphAny-SF 在四个公平迁移数据集上的运行入口。

协议：
1. 在 source 图上训练 GraphAny。
2. 保存模型参数和 source 上拟合得到的各 channel LinearGNN 权重。
3. target 阶段只加载这些 source-derived artifact，并用 target 无标签传播
   特征做测试；target 标签只用于最终指标计算，不参与输入构造。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for import_path in (SCRIPT_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from graphany.sfda_data import FairGraphDataset
from graphany.model import GraphAny
from visualization.export_utils import save_visualization_embeddings


DATASET_PAIRS = {
    "bailA": ("bailA_2", "bailA_1"),
    "germanA": ("germanA_2", "germanA_1"),
    "pokec": ("pokec_z", "pokec_n"),
    "syn": ("syn-2", "syn-1"),
}

N_RUNS = 5
RESULT_DIR = SCRIPT_DIR / "results"
RUN_DIR = RESULT_DIR / "runs"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
LOG_DIR = RESULT_DIR / "logs"
SUMMARY_FILE = RESULT_DIR / "graphany_sf_summary.md"


def set_seed(seed):
    """固定随机性，保证 5 次运行只由 run seed 区分。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_cfg(args):
    """构造 GraphAny 原实验需要的数据配置。"""
    feat_channels = args.feat_chn.split("+")
    pred_channels = args.pred_chn.split("+")
    return argparse.Namespace(
        add_self_loop=args.add_self_loop,
        to_bidirected=args.to_bidirected,
        n_hops=args.n_hops,
        feat_chn=args.feat_chn,
        pred_chn=args.pred_chn,
        feat_channels=feat_channels,
        pred_channels=pred_channels,
        entropy=args.entropy,
        n_per_label_examples=args.n_per_label_examples,
    )


def resolve_run_path(path):
    """相对路径统一解释为 GraphAny 目录下的路径。"""
    path = Path(path)
    return path if path.is_absolute() else SCRIPT_DIR / path


def build_model(cfg, args, device):
    return GraphAny(
        n_hidden=args.n_hidden,
        feat_channels=cfg.feat_channels,
        pred_channels=cfg.pred_channels,
        att_temperature=args.attn_temp,
        entropy=args.entropy,
        n_mlp_layer=args.n_mlp_layer,
    ).to(device)


def get_device(cuda_id):
    if cuda_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(cuda_id)
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def sample_visible_nodes(train_indices, batch_nodes):
    """当前 batch 节点不可见，沿用 GraphAny 原训练协议。"""
    return train_indices[~torch.isin(train_indices, batch_nodes)]


def train_one_epoch(model, optimizer, criterion, ds, batch_size, device, limit_batches):
    model.train()
    train_idx = ds.train_indices[torch.randperm(len(ds.train_indices))]
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(train_idx), batch_size):
        batch_cpu = train_idx[start : start + batch_size]
        visible = sample_visible_nodes(ds.train_indices, batch_cpu)
        if len(visible) < len(batch_cpu):
            visible = torch.cat([visible, batch_cpu[: len(batch_cpu) // 2]])

        input_logits = ds.compute_channel_logits(
            ds.features,
            visible,
            sample=True,
            device=device,
        )
        batch = batch_cpu.to(device)
        preds, _ = model(
            {channel: logits[batch] for channel, logits in input_logits.items()},
            dist=None,
        )
        loss = criterion(preds, ds.label[batch_cpu].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if limit_batches > 0 and n_batches >= limit_batches:
            break

    return total_loss / max(n_batches, 1)


def safe_mean(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else float("nan")


def fair_metric(pred, labels, sens):
    """计算 DP/EO，返回百分制数值。"""
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    sens = np.asarray(sens)

    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    dp = abs(safe_mean(pred[idx_s0]) - safe_mean(pred[idx_s1]))
    eo = abs(safe_mean(pred[idx_s0_y1]) - safe_mean(pred[idx_s1_y1]))
    return dp * 100.0, eo * 100.0


def eval_indices(ds, split):
    if split == "train":
        return ds.train_mask.nonzero().view(-1)
    if split == "val":
        return ds.val_mask.nonzero().view(-1)
    if split == "test":
        return ds.test_mask.nonzero().view(-1)
    if split == "all":
        # 与 SFFGNN 对齐：all 指所有有监督划分覆盖的节点。
        # Pokec 中 label=-1 的节点不会进入 train/val/test，因此这里会被排除。
        return (ds.train_mask | ds.val_mask | ds.test_mask).nonzero().view(-1)
    raise ValueError(f"Unknown eval split: {split}")


@torch.no_grad()
def evaluate_target(model, ds, channel_weights, cfg, args, device):
    """用 source-derived channel 权重评估 target，不使用 target 训练标签。"""
    model.eval()
    idx = eval_indices(ds, args.eval_split)

    all_pred, all_prob, all_label, all_sens = [], [], [], []
    channels = set(cfg.feat_channels + cfg.pred_channels)
    weights = {channel: channel_weights[channel].to(device) for channel in channels}

    for start in range(0, len(idx), args.eval_batch):
        batch_cpu = idx[start : start + args.eval_batch]
        logit_slice = {
            channel: ds.features[channel][batch_cpu].to(device) @ weights[channel]
            for channel in channels
        }
        preds, _ = model(logit_slice, dist=None)
        all_pred.append(preds.argmax(dim=-1).cpu())
        all_prob.append(F.softmax(preds, dim=-1)[:, 1].cpu())
        all_label.append(ds.label[batch_cpu].cpu())
        all_sens.append(ds.sens_labels[batch_cpu].cpu())

    y_pred = torch.cat(all_pred).numpy()
    y_prob = torch.cat(all_prob).numpy()
    y_true = torch.cat(all_label).numpy()
    s_true = torch.cat(all_sens).numpy().astype(int)

    acc = accuracy_score(y_true, y_pred) * 100.0
    auc = roc_auc_score(y_true, y_prob) * 100.0 if len(set(y_true)) == 2 else float("nan")
    dp, eo = fair_metric(y_pred, y_true, s_true)
    return {"Acc": acc, "AUC": auc, "DP": dp, "EO": eo}


@torch.no_grad()
def export_target_embeddings(model, ds, channel_weights, cfg, args, device):
    model.eval()
    all_mask = (
        ds.train_mask | ds.val_mask | ds.test_mask
    ) & (ds.label >= 0)
    idx = all_mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        raise ValueError("Target all split has no valid labels for visualization export")

    channels = set(cfg.feat_channels + cfg.pred_channels)
    weights = {channel: channel_weights[channel].to(device) for channel in channels}
    representations = []
    for start in range(0, len(idx), args.eval_batch):
        batch_cpu = idx[start : start + args.eval_batch]
        logit_slice = {
            channel: ds.features[channel][batch_cpu].to(device) @ weights[channel]
            for channel in channels
        }
        _, _, batch_representation = model(
            logit_slice,
            dist=None,
            return_representation=True,
        )
        representations.append(batch_representation.cpu())

    target_features = torch.cat(representations, dim=0).numpy()
    feat_path, labels_path = save_visualization_embeddings(
        REPO_ROOT / "visualization" / "embeddings",
        "GraphAny",
        args.dataset,
        target_features,
        y=ds.label[idx].detach().cpu().numpy(),
        sens=ds.sens_labels[idx].detach().cpu().numpy(),
    )
    print(f"[saved] {feat_path}", flush=True)
    print(f"[saved] {labels_path}", flush=True)


def checkpoint_path(dataset, run_idx):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"GraphAny-SF_{dataset}_run{run_idx}.pt"


def run_result_path(dataset, run_idx):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{dataset}_run{run_idx}.json"


def dataset_result_path(dataset):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{dataset}.txt"


def save_source_artifact(path, model, channel_weights, cfg, args, dataset, run_idx):
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "channel_weights": {
                k: v.detach().cpu() for k, v in channel_weights.items()
            },
            "cfg": vars(cfg),
            "model_args": {
                "n_hidden": args.n_hidden,
                "attn_temp": args.attn_temp,
                "entropy": args.entropy,
                "n_mlp_layer": args.n_mlp_layer,
            },
            "dataset": dataset,
            "run_idx": run_idx,
            "seed": args.seed + run_idx,
        },
        path,
    )


def write_run_result(dataset, run_idx, metrics, args):
    path = run_result_path(dataset, run_idx)
    payload = {
        "method": "GraphAny-SF",
        "dataset": dataset,
        "run_idx": run_idx,
        "seed": args.seed + run_idx,
        "eval_split": args.eval_split,
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def run_single(args):
    source_name, target_name = DATASET_PAIRS[args.dataset]
    set_seed(args.seed + args.run_idx)
    cfg = build_cfg(args)
    device = get_device(args.cuda)

    cache_dir = resolve_run_path(args.cache_dir) / args.dataset / f"run{args.run_idx}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[GraphAny-SF] dataset={args.dataset} run={args.run_idx} "
        f"source={source_name} target={target_name} device={device}",
        flush=True,
    )

    source_ds = FairGraphDataset(
        source_name,
        cfg,
        str(cache_dir),
        train_batch_size=args.train_batch,
        val_test_batch_size=args.eval_batch,
        preprocess_device=device,
    )
    model = build_model(cfg, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model,
            optimizer,
            criterion,
            source_ds,
            args.train_batch,
            device,
            args.limit_train_batches,
        )
        if args.verbose and (epoch + 1) % args.print_every == 0:
            print(
                f"[{args.dataset} run{args.run_idx}] "
                f"epoch={epoch + 1}/{args.epochs} loss={loss:.6f}",
                flush=True,
            )

    # 只保存 source 训练得到的参数和 source-derived channel 权重。
    channel_weights = source_ds.fit_channel_weights(
        source_ds.features,
        source_ds.train_indices,
        bootstrap=False,
    )
    ckpt_path = checkpoint_path(args.dataset, args.run_idx)
    save_source_artifact(
        ckpt_path,
        model,
        channel_weights,
        cfg,
        args,
        args.dataset,
        args.run_idx,
    )

    del source_ds
    del model
    torch.cuda.empty_cache()

    artifact = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg, args, device)
    model.load_state_dict(
        {k: v.to(device) for k, v in artifact["model_state"].items()}
    )
    channel_weights = artifact["channel_weights"]

    target_ds = FairGraphDataset(
        target_name,
        cfg,
        str(cache_dir),
        train_batch_size=args.train_batch,
        val_test_batch_size=args.eval_batch,
        preprocess_device=device,
    )
    metrics = evaluate_target(model, target_ds, channel_weights, cfg, args, device)
    if (
        args.save_visualization_embeddings
        and args.run_idx == args.visualization_run_idx
    ):
        export_target_embeddings(model, target_ds, channel_weights, cfg, args, device)
    out_path = write_run_result(args.dataset, args.run_idx, metrics, args)
    print(f"[GraphAny-SF] run result saved: {out_path}", flush=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


def fmt_mean_std(values):
    arr = np.asarray(values, dtype=float)
    return f"{np.nanmean(arr):.2f}±{np.nanstd(arr):.2f}"


def aggregate_dataset(dataset):
    rows = []
    for run_idx in range(N_RUNS):
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
        f.write("|Dataset|ACC|AUC|DP|EO|\n")
        f.write("|---|---|---|---|---|\n")
        f.write(
            f"|{dataset}|{summary['Acc']}|{summary['AUC']}|"
            f"{summary['DP']}|{summary['EO']}|\n"
        )
    return summary


def aggregate_all(datasets):
    summaries = {}
    for dataset in datasets:
        summary = aggregate_dataset(dataset)
        if summary is not None:
            summaries[dataset] = summary

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("|Dataset|ACC|AUC|DP|EO|\n")
        f.write("|---|---|---|---|---|\n")
        for dataset in datasets:
            if dataset not in summaries:
                continue
            row = summaries[dataset]
            f.write(
                f"|{dataset}|{row['Acc']}|{row['AUC']}|"
                f"{row['DP']}|{row['EO']}|\n"
            )
    return SUMMARY_FILE


def parse_gpu_ids(gpus):
    ids = []
    for item in gpus.split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return ids or [-1]


def launch_parallel(args):
    """父进程调度所有数据集和 run，尽量填满可用 GPU。"""
    if args.save_visualization_embeddings and not 0 <= args.visualization_run_idx < N_RUNS:
        raise ValueError(f"visualization_run_idx must be in [0, {N_RUNS - 1}]")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(dataset, run_idx) for dataset in args.datasets for run_idx in range(N_RUNS)]
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
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--dataset",
            dataset,
            "--run_idx",
            str(run_idx),
            "--cuda",
            str(gpu_id),
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--train_batch",
            str(args.train_batch),
            "--eval_batch",
            str(args.eval_batch),
            "--limit_train_batches",
            str(args.limit_train_batches),
            "--n_hidden",
            str(args.n_hidden),
            "--n_mlp_layer",
            str(args.n_mlp_layer),
            "--entropy",
            str(args.entropy),
            "--attn_temp",
            str(args.attn_temp),
            "--feat_chn",
            args.feat_chn,
            "--pred_chn",
            args.pred_chn,
            "--n_hops",
            str(args.n_hops),
            "--n_per_label_examples",
            str(args.n_per_label_examples),
            "--eval_split",
            args.eval_split,
            "--cache_dir",
            args.cache_dir,
            "--print_every",
            str(args.print_every),
        ]
        if args.add_self_loop:
            cmd.append("--add_self_loop")
        if not args.to_bidirected:
            cmd.append("--no_to_bidirected")
        if args.verbose:
            cmd.append("--verbose")
        if args.save_visualization_embeddings and run_idx == args.visualization_run_idx:
            cmd.extend([
                "--save_visualization_embeddings",
                "--visualization_run_idx",
                str(args.visualization_run_idx),
            ])

        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
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

            dataset_runs = {(info["dataset"], idx) for idx in range(N_RUNS)}
            if dataset_runs.issubset(completed):
                summary = aggregate_dataset(info["dataset"])
                print(f"[Dataset summary] {info['dataset']}: {summary}")

    out = aggregate_all(args.datasets)
    print(f"[All done] summary saved: {out}")


def get_args():
    parser = argparse.ArgumentParser(description="GraphAny-SF fair benchmark runner")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dataset", choices=list(DATASET_PAIRS.keys()), default="bailA")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bailA", "germanA", "pokec", "syn"],
        choices=list(DATASET_PAIRS.keys()),
    )
    parser.add_argument("--run_idx", type=int, default=0)
    parser.add_argument("--save_visualization_embeddings", action="store_true")
    parser.add_argument("--visualization_run_idx", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0,1,2,4,5,6,7")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--train_batch", type=int, default=128)
    parser.add_argument("--eval_batch", type=int, default=100000)
    parser.add_argument("--limit_train_batches", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_mlp_layer", type=int, default=2)
    parser.add_argument("--entropy", type=float, default=1.0)
    parser.add_argument("--attn_temp", type=float, default=5.0)
    parser.add_argument("--feat_chn", type=str, default="X+L1+L2+H1+H2")
    parser.add_argument("--pred_chn", type=str, default="X+L1+L2")
    parser.add_argument("--n_hops", type=int, default=2)
    parser.add_argument("--n_per_label_examples", type=int, default=5)
    parser.add_argument("--add_self_loop", action="store_true")
    parser.add_argument("--to_bidirected", dest="to_bidirected", action="store_true")
    parser.add_argument("--no_to_bidirected", dest="to_bidirected", action="store_false")
    parser.set_defaults(to_bidirected=True)

    parser.add_argument("--eval_split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--cache_dir", type=str, default="./data_cache/fair_dg")
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = get_args()
    if cli_args.aggregate_only:
        out_file = aggregate_all(cli_args.datasets)
        print(f"[Aggregate] summary saved: {out_file}")
    elif cli_args.worker:
        run_single(cli_args)
    else:
        launch_parallel(cli_args)
