# -*- coding: utf-8 -*-
"""
DGSDA-SF: 将标准 UGDA 方法 DGSDA 改造成 source-free domain adaptation 对比基线。

改造原则：
1. 阶段一只在源域图上用 L_source 预训练，并保存 final-epoch checkpoint。
2. 阶段二只加载目标域图，删除需要源/目标同时前向的 L_mmd。
3. 阶段二冻结除目标域 BernNet 系数 theta^T 外的所有参数，只优化
   L_SFDA = alpha * L_align_SF + gamma * L_target。
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
import torch.nn.functional as F

from models import BernNet

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError:  # 服务器若缺少 sklearn，AUC 会被置为 NaN，Acc/DP/EO 仍可运行。
    accuracy_score = None
    roc_auc_score = None

sys.path.append(str(Path(__file__).resolve().parents[2]))

from visualization.export_utils import save_visualization_embeddings


N_RUNS = 5
REPO_ROOT = Path(__file__).resolve().parents[2]
SFFGNN_DIR = REPO_ROOT / "SFFGNN"
RESULT_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
LOG_DIR = RESULT_DIR / "logs"

DATASET_ID_MAP = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}


def load_sffgnn_dataset_api():
    """按 SFFGNN/dataset.py 的原始路径与逻辑加载数据接口。"""
    if str(SFFGNN_DIR) not in sys.path:
        sys.path.insert(0, str(SFFGNN_DIR))
    from EMBER.dataset import get_dataset
    return get_dataset


def seed_everything(seed):
    """固定随机种子，保证 3 次运行只由 run seed 控制。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def entropy_minimization_loss(output):
    """沿用原始 DGSDA 的目标域熵最小化损失。"""
    probs = F.softmax(output, dim=1)
    log_probs = F.log_softmax(output, dim=1)
    class_mass = torch.sum(probs, dim=0)
    return -torch.sum(probs * log_probs / (class_mass / torch.sum(class_mass)), dim=1).mean()


def align_sf_loss(theta_s_fixed, theta_t):
    """只用预存 theta^S 与当前 theta^T 计算 source-free 对齐损失。"""
    return (
        torch.sum(torch.abs(theta_s_fixed - theta_t))
        + torch.sum(torch.abs(theta_s_fixed))
        + torch.sum(torch.abs(theta_t))
    )


def get_device(cuda_id):
    if cuda_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(cuda_id)
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def load_graph(dataset, domain_id, device=None):
    """复用 SFFGNN 的 get_dataset，确保数据路径与处理逻辑完全一致。"""
    get_dataset = load_sffgnn_dataset_api()
    data_args = SimpleNamespace(dataset=dataset)
    data = get_dataset(data_args, domain_id)
    return data.to(device) if device is not None else data


def build_model(data, args, device):
    """DGSDA 原始 BernNet：属性编码器 h_X=lin1，分类器=lin2，prop1/prop2 为源/目标 BernNet。"""
    num_features = data.x.shape[1]
    # pokec 中存在 y=-1 的未标注节点；类别数只由有效标签 {0, 1} 决定。
    valid_labels = data.y[data.y >= 0]
    num_classes = int(valid_labels.max().item()) + 1
    return BernNet(
        num_features,
        args.hidden,
        num_classes,
        args.dropout_ratio,
        args.dp_ratio,
        args.K,
    ).to(device)


def checkpoint_path(dataset, run_idx):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"DGSDA-SF_{dataset}_run{run_idx}_final.pt"


def run_result_path(dataset, run_idx):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{dataset}_run{run_idx}_result.json"


def dataset_result_path(dataset):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{dataset}_DGSDA-SF_results.txt"


def summary_path():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / "DGSDA-SF_summary.md"


def save_checkpoint(path, model, theta_s_fixed, dataset, source_id, run_idx, seed, args):
    """保存阶段二所需的完整模型参数和固定源域 Bernstein 系数。"""
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "theta_s_fixed": theta_s_fixed.detach().cpu(),
            "dataset": dataset,
            "source_id": source_id,
            "run_idx": run_idx,
            "seed": seed,
            "model_args": {
                "hidden": args.hidden,
                "dropout_ratio": args.dropout_ratio,
                "dp_ratio": args.dp_ratio,
                "K": args.K,
            },
        },
        path,
    )


def stage1_pretrain_source(dataset, source_id, run_idx, args, device):
    """阶段一：仅访问源域图，使用 L_source 单独预训练。"""
    source_data = load_graph(dataset, source_id, device)
    model = build_model(source_data, args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.source_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(source_data, True)
        loss = F.cross_entropy(output[source_data.train_mask], source_data.y[source_data.train_mask])
        loss.backward()
        optimizer.step()

        if args.verbose and (epoch + 1) % args.print_every == 0:
            print(f"[{dataset} run{run_idx}] stage1 epoch={epoch + 1} L_source={loss.item():.6f}")

    ckpt_path = checkpoint_path(dataset, run_idx)
    theta_s_fixed = model.prop1.temp.detach().clone()
    save_checkpoint(ckpt_path, model, theta_s_fixed, dataset, source_id, run_idx, args.seed + run_idx, args)

    # 阶段一结束后显式释放源域图；阶段二函数不会接收 source_data。
    del source_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path


def stage2_adapt_target(dataset, target_id, run_idx, ckpt_path, args, device):
    """阶段二：只加载目标域图，冻结除 theta^T 外的所有参数。"""
    target_data = load_graph(dataset, target_id, device)
    model = build_model(target_data, args, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["model_state"].items()})
    theta_s_fixed = ckpt["theta_s_fixed"].to(device)

    for param in model.parameters():
        param.requires_grad_(False)
    model.prop2.temp.requires_grad_(True)

    # 阶段二严格只优化 L_SFDA，不额外引入 optimizer weight_decay 正则。
    optimizer = torch.optim.Adam([model.prop2.temp], lr=args.lr)

    for epoch in range(args.target_epochs):
        model.train()
        optimizer.zero_grad()
        theta_loss = align_sf_loss(theta_s_fixed, model.prop2.temp)
        target_output = model(target_data, False)
        target_loss = entropy_minimization_loss(target_output)
        total_loss = args.alpha * theta_loss + args.gamma * target_loss
        total_loss.backward()
        optimizer.step()

        if args.verbose and (epoch + 1) % args.print_every == 0:
            print(
                f"[{dataset} run{run_idx}] stage2 epoch={epoch + 1} "
                f"L_align_SF={theta_loss.item():.6f} L_target={target_loss.item():.6f}"
            )

    if args.save_visualization_embeddings and run_idx == args.visualization_run_idx:
        model.eval()
        with torch.no_grad():
            target_features = model.get_props(
                target_data.x,
                target_data.edge_index,
                is_source_domain=False,
            )
        all_mask = (
            target_data.train_mask | target_data.val_mask | target_data.test_mask
        ) & (target_data.y >= 0)
        feat_path, labels_path = save_visualization_embeddings(
            REPO_ROOT / "visualization" / "embeddings",
            "DGSDA",
            dataset,
            target_features[all_mask].detach().cpu().numpy(),
            y=target_data.y[all_mask].detach().cpu().numpy(),
            sens=target_data.sens_labels[all_mask].detach().cpu().numpy(),
        )
        print(f"[saved] {feat_path}")
        print(f"[saved] {labels_path}")

    return evaluate_target(model, target_data)


def safe_mean(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else float("nan")


def fair_metric(pred, labels, sens):
    """按 SFFGNN 口径计算 DP/EO，并转为百分制。"""
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


def evaluate_target(model, target_data):
    """评估目标域结果；按 SFFGNN 的 all split 口径排除 pokec 的 y=-1 节点。"""
    model.eval()
    with torch.no_grad():
        output = model(target_data, False)
        # SFFGNN 的 all split 是 train/val/test 三个 mask 的并集。
        # 对 pokec，y=-1 节点没有进入这些 mask，因此不会参与 Acc/AUC/DP/EO。
        eval_mask = target_data.train_mask | target_data.val_mask | target_data.test_mask
        eval_mask = eval_mask & (target_data.y >= 0)
        prob = F.softmax(output, dim=1)[:, 1][eval_mask].detach().cpu().numpy()
        pred = output.argmax(dim=1)[eval_mask].detach().cpu().numpy()
        labels = target_data.y[eval_mask].detach().cpu().numpy()
        sens = target_data.sens_labels[eval_mask].detach().cpu().numpy()

    if accuracy_score is not None:
        acc = accuracy_score(labels, pred) * 100.0
    else:
        acc = float((pred == labels).mean() * 100.0)

    if roc_auc_score is not None and len(set(labels.tolist())) == 2:
        auc = roc_auc_score(labels, prob) * 100.0
    else:
        auc = float("nan")

    dp, eo = fair_metric(pred, labels, sens)
    return {"Acc": acc, "AUC": auc, "DP": dp, "EO": eo}


def write_run_result(dataset, run_idx, metrics, args):
    path = run_result_path(dataset, run_idx)
    payload = {
        "method": "DGSDA-SF",
        "dataset": dataset,
        "run_idx": run_idx,
        "seed": args.seed + run_idx,
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def run_single(args):
    source_id, target_id = DATASET_ID_MAP[args.dataset]
    seed_everything(args.seed + args.run_idx)
    device = get_device(args.cuda)

    print(
        f"[DGSDA-SF] dataset={args.dataset} run={args.run_idx} "
        f"source={source_id} target={target_id} device={device}"
    )
    ckpt = stage1_pretrain_source(args.dataset, source_id, args.run_idx, args, device)
    metrics = stage2_adapt_target(args.dataset, target_id, args.run_idx, ckpt, args, device)
    out_path = write_run_result(args.dataset, args.run_idx, metrics, args)
    print(f"[DGSDA-SF] result saved: {out_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


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
        f.write(f"Dataset: {dataset}\n")
        f.write("Method: DGSDA-SF\n")
        f.write("|Dataset|ACC|AUC|DP|EO|\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"|{dataset}|{summary['Acc']}|{summary['AUC']}|{summary['DP']}|{summary['EO']}|\n")
        f.write("\nRaw runs:\n")
        for idx, row in enumerate(rows):
            f.write(
                f"run{idx}: Acc={row['Acc']:.4f}, AUC={row['AUC']:.4f}, "
                f"DP={row['DP']:.4f}, EO={row['EO']:.4f}\n"
            )
    return summary


def aggregate_all(datasets):
    summaries = {}
    for dataset in datasets:
        summary = aggregate_dataset(dataset)
        if summary is not None:
            summaries[dataset] = summary

    out = summary_path()
    with open(out, "w", encoding="utf-8") as f:
        f.write("|Dataset|ACC|AUC|DP|EO|\n")
        f.write("|---|---|---|---|---|\n")
        for dataset in datasets:
            if dataset not in summaries:
                continue
            row = summaries[dataset]
            f.write(f"|{dataset}|{row['Acc']}|{row['AUC']}|{row['DP']}|{row['EO']}|\n")
    return out


def parse_gpu_ids(gpus):
    ids = []
    for item in gpus.split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return ids or [-1]


def launch_parallel(args):
    """父进程：将 4 个数据集 x 3 次运行分配到可用 GPU 上并行执行。"""
    if args.save_visualization_embeddings and not 0 <= args.visualization_run_idx < N_RUNS:
        raise ValueError(f"visualization_run_idx must be in [0, {N_RUNS - 1}]")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    tasks = [(dataset, run_idx) for dataset in datasets for run_idx in range(N_RUNS)]
    gpu_ids = parse_gpu_ids(args.gpus)
    running = {}
    completed = set()

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
            "--cuda",
            str(gpu_id),
            "--seed",
            str(args.seed),
            "--lr",
            str(args.lr),
            "--wd",
            str(args.wd),
            "--hidden",
            str(args.hidden),
            "--K",
            str(args.K),
            "--dropout_ratio",
            str(args.dropout_ratio),
            "--dp_ratio",
            str(args.dp_ratio),
            "--source_epochs",
            str(args.source_epochs),
            "--target_epochs",
            str(args.target_epochs),
            "--alpha",
            str(args.alpha),
            "--gamma",
            str(args.gamma),
            "--print_every",
            str(args.print_every),
        ]
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
        running[proc] = {"dataset": dataset, "run_idx": run_idx, "gpu": gpu_id, "log": log_file}
        print(f"[Launch] {dataset} run{run_idx} -> GPU {gpu_id}, log={log_path}")

    task_queue = list(tasks)
    free_gpus = list(gpu_ids)
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
                    f"see {LOG_DIR / (info['dataset'] + '_run' + str(info['run_idx']) + '.log')}"
                )

            completed.add((info["dataset"], info["run_idx"]))
            print(f"[Done] {info['dataset']} run{info['run_idx']}")

            dataset_runs = {(info["dataset"], idx) for idx in range(N_RUNS)}
            if dataset_runs.issubset(completed):
                summary = aggregate_dataset(info["dataset"])
                print(f"[Dataset summary] {info['dataset']}: {summary}")

    out = aggregate_all(datasets)
    print(f"[All done] summary saved: {out}")


def get_args():
    parser = argparse.ArgumentParser(description="DGSDA-SF source-free baseline runner")
    parser.add_argument("--worker", action="store_true", help="run one dataset/run worker")
    parser.add_argument("--aggregate_only", action="store_true", help="only aggregate existing run json files")
    parser.add_argument("--dataset", type=str, default="syn", choices=list(DATASET_ID_MAP.keys()))
    parser.add_argument("--datasets", nargs="+", default=["bailA", "germanA", "pokec", "syn"])
    parser.add_argument("--run_idx", type=int, default=0)
    parser.add_argument("--save_visualization_embeddings", action="store_true")
    parser.add_argument("--visualization_run_idx", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0,1,2,4,5,6,7")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1111)

    # DGSDA 原始超参数；alpha/gamma 沿用 main.py 默认设置。
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--dropout_ratio", type=float, default=0.3)
    parser.add_argument("--dp_ratio", type=float, default=0.3)
    parser.add_argument("--source_epochs", type=int, default=100)
    parser.add_argument("--target_epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--print_every", type=int, default=20)
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
