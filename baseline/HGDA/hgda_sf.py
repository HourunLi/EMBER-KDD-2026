# -*- coding: utf-8 -*-
"""
HGDA-SF 在 SFFGNN 四个数据集上的运行入口。

实验流程：
1. 阶段一只加载源域图，用 HGDA 原始分类损失训练到 final epoch。
2. 保存 final-epoch 模型参数和三个滤波分支的源域类条件原型。
3. 阶段二释放源域图，只加载目标域图、final checkpoint 和源域原型做无源适配。
4. 每个数据集默认运行 3 次，输出 Acc/AUC/DP/EO 的 mean±std。
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
from torch_geometric.utils import add_self_loops, degree

try:
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError:
    accuracy_score = None
    roc_auc_score = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SFFGNN_DIR = REPO_ROOT / "SFFGNN"
SCRIPT_DIR = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SFFGNN_DIR) not in sys.path:
    sys.path.insert(0, str(SFFGNN_DIR))

from EMBER.dataset import get_dataset
from visualization.export_utils import save_visualization_embeddings


DATASET_ID_MAP = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

METHOD_NAME = "HGDA-SF"
RESULT_DIR = SCRIPT_DIR / "results"
RUN_DIR = RESULT_DIR / "runs"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
LOG_DIR = RESULT_DIR / "logs"
SUMMARY_FILE = RESULT_DIR / "HGDA-SF_summary.md"


def seed_everything(seed):
    """固定随机种子，保证不同 run 只由 run seed 区分。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    """复用 SFFGNN.dataset.get_dataset，保持数据路径、特征归一化和 mask 构造完全一致。"""
    data = get_dataset(SimpleNamespace(dataset=dataset), domain_id)
    return data.to(device)


def valid_label_mask(data):
    """pokec 含 y=-1 的未标注节点，训练和评估均按 SFFGNN mask 排除。"""
    return data.y >= 0


def train_mask(data):
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
        raise ValueError(f"Unknown split: {split}")
    return mask & valid_label_mask(data)


def infer_num_classes(data):
    labels = data.y[valid_label_mask(data)]
    return int(labels.max().item()) + 1


def compute_normalized_adjacency(edge_index, num_nodes):
    """计算归一化邻接矩阵 Â = D^(-1/2) A D^(-1/2)。"""
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float).to(edge_index.device)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        (num_nodes, num_nodes),
        device=edge_index.device,
    ).coalesce()


def compute_normalized_laplacian(edge_index, num_nodes):
    """计算归一化拉普拉斯矩阵 L̃ = I - Â。"""
    adj = compute_normalized_adjacency(edge_index, num_nodes)
    diag = torch.arange(num_nodes, device=edge_index.device)
    identity_index = torch.stack([diag, diag], dim=0)
    indices = torch.cat([identity_index, adj.indices()], dim=1)
    values = torch.cat([
        torch.ones(num_nodes, device=edge_index.device),
        -adj.values(),
    ])
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        device=edge_index.device,
    ).coalesce()


class SpectralFilter(nn.Module):
    """图谱滤波器基类。"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.weight = nn.Linear(input_dim, hidden_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, filter_matrix):
        filtered = torch.sparse.mm(filter_matrix, x)
        transformed = self.weight(filtered)
        return F.relu(self.alpha * transformed)


class HomophilicFilter(SpectralFilter):
    """同质性低通滤波器。"""
    def forward(self, x, adj_normalized):
        return super().forward(x, adj_normalized)


class FullPassFilter(SpectralFilter):
    """全通滤波器，只保留属性变换。"""
    def forward(self, x, num_nodes):
        del num_nodes
        transformed = self.weight(x)
        return F.relu(self.alpha * transformed)


class HeterophilicFilter(SpectralFilter):
    """异质性高通滤波器。"""
    def forward(self, x, laplacian_normalized):
        return super().forward(x, laplacian_normalized)


class DomainAlignmentModel(nn.Module):
    """保留 HGDA 原始三滤波分支和可学习组合权重。"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.homophilic_filters = nn.ModuleList()
        self.fullpass_filters = nn.ModuleList()
        self.heterophilic_filters = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.homophilic_filters.append(HomophilicFilter(in_dim, hidden_dim))
            self.fullpass_filters.append(FullPassFilter(in_dim, hidden_dim))
            self.heterophilic_filters.append(HeterophilicFilter(in_dim, hidden_dim))

        self.weight_homo = nn.Parameter(torch.tensor(1.0))
        self.weight_full = nn.Parameter(torch.tensor(1.0))
        self.weight_hetero = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, num_nodes):
        adj_normalized = compute_normalized_adjacency(edge_index, num_nodes)
        laplacian_normalized = compute_normalized_laplacian(edge_index, num_nodes)

        h_homo = x
        h_full = x
        h_hetero = x
        for i in range(self.num_layers):
            h_homo = self.homophilic_filters[i](h_homo, adj_normalized)
            h_full = self.fullpass_filters[i](h_full, num_nodes)
            h_hetero = self.heterophilic_filters[i](h_hetero, laplacian_normalized)

        combined = (
            self.weight_homo * h_homo
            + self.weight_full * h_full
            + self.weight_hetero * h_hetero
        )
        return combined, h_homo, h_full, h_hetero


def build_model(data, args, device):
    return DomainAlignmentModel(
        input_dim=data.x.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=infer_num_classes(data),
        num_layers=args.num_layers,
    ).to(device)


def entropy_minimization_loss(logits):
    """目标域熵最小化损失 L_T。"""
    probs = F.softmax(logits, dim=1).clamp(min=1e-8)
    return -(probs * probs.log()).sum(dim=1).mean()


def compute_class_conditional_prototypes(h_homo, h_full, h_hetero, labels, mask, num_classes):
    """按源域训练节点标签计算三个滤波分支的类条件原型。"""
    prototypes = {"homo": [], "full": [], "hetero": []}
    for class_id in range(num_classes):
        class_mask = mask & (labels == class_id)
        if class_mask.sum().item() == 0:
            prototypes["homo"].append(torch.zeros(h_homo.size(1), device=h_homo.device))
            prototypes["full"].append(torch.zeros(h_full.size(1), device=h_full.device))
            prototypes["hetero"].append(torch.zeros(h_hetero.size(1), device=h_hetero.device))
            continue

        prototypes["homo"].append(h_homo[class_mask].mean(dim=0))
        prototypes["full"].append(h_full[class_mask].mean(dim=0))
        prototypes["hetero"].append(h_hetero[class_mask].mean(dim=0))

    return {name: torch.stack(values, dim=0).detach() for name, values in prototypes.items()}


def prototype_kl_loss(target_mean, source_prototype):
    """沿用 HGDA 原始 KL 写法，在隐藏维 softmax 分布上比较 target 均值和 source 原型。"""
    return F.kl_div(
        F.log_softmax(target_mean.unsqueeze(0), dim=1),
        F.softmax(source_prototype.unsqueeze(0), dim=1),
        reduction="batchmean",
    )


def source_free_alignment_loss(tgt_homo, tgt_full, tgt_hetero, tgt_logits, prototypes):
    """用目标伪标签和源域原型计算 L_H^SF。"""
    with torch.no_grad():
        pseudo_labels = tgt_logits.argmax(dim=1)

    total_loss = tgt_logits.new_tensor(0.0)
    num_nodes = tgt_logits.size(0)
    num_classes = prototypes["homo"].size(0)

    for class_id in range(num_classes):
        class_mask = pseudo_labels == class_id
        class_count = class_mask.sum()
        if class_count.item() == 0:
            continue

        class_prior = class_count.float() / float(num_nodes)
        class_loss = (
            prototype_kl_loss(tgt_homo[class_mask].mean(dim=0), prototypes["homo"][class_id])
            + prototype_kl_loss(tgt_hetero[class_mask].mean(dim=0), prototypes["hetero"][class_id])
            + prototype_kl_loss(tgt_full[class_mask].mean(dim=0), prototypes["full"][class_id])
        )
        total_loss = total_loss + class_prior * class_loss

    return total_loss


def checkpoint_path(dataset, run_idx):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"HGDA-SF_{dataset}_run{run_idx}_final.pt"


def run_result_path(dataset, run_idx):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{dataset}_run{run_idx}.json"


def dataset_result_path(dataset):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{dataset}_HGDA-SF_results.txt"


def save_checkpoint(path, model, prototypes, dataset, source_id, run_idx, seed, args):
    """保存 final-epoch checkpoint 和源域原型，不保存源域原始数据。"""
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "prototypes": {k: v.detach().cpu() for k, v in prototypes.items()},
            "dataset": dataset,
            "source_id": source_id,
            "run_idx": run_idx,
            "seed": seed,
            "model_args": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "output_dim": model.output_dim,
                "num_layers": args.num_layers,
            },
            "train_args": {
                "source_lr": args.source_lr,
                "target_lr": args.target_lr,
                "source_epochs": args.source_epochs,
                "target_epochs": args.target_epochs,
                "alpha": args.alpha,
            },
        },
        path,
    )


def load_checkpoint(path, device):
    artifact = torch.load(path, map_location=device)
    model_args = artifact["model_args"]
    model = DomainAlignmentModel(
        input_dim=model_args["input_dim"],
        hidden_dim=model_args["hidden_dim"],
        output_dim=model_args["output_dim"],
        num_layers=model_args["num_layers"],
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in artifact["model_state"].items()})
    prototypes = {k: v.to(device) for k, v in artifact["prototypes"].items()}
    return model, prototypes


def stage1_pretrain_source(dataset, source_id, run_idx, args, device):
    """阶段一：仅访问源域图，使用源域训练 mask 上的交叉熵。"""
    source_data = load_graph(dataset, source_id, device)
    model = build_model(source_data, args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.source_lr)
    source_train_mask = train_mask(source_data)

    for epoch in range(args.source_epochs):
        model.train()
        optimizer.zero_grad()
        src_combined, _, _, _ = model(source_data.x, source_data.edge_index, source_data.x.size(0))
        src_logits = model.classifier(src_combined)
        loss = F.cross_entropy(src_logits[source_train_mask], source_data.y[source_train_mask])
        loss.backward()
        optimizer.step()

        if args.verbose and (epoch + 1) % args.print_every == 0:
            print(f"[{dataset} run{run_idx}] 阶段一 epoch={epoch + 1} L_source={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        _, src_homo, src_full, src_hetero = model(
            source_data.x, source_data.edge_index, source_data.x.size(0)
        )
        prototypes = compute_class_conditional_prototypes(
            src_homo,
            src_full,
            src_hetero,
            source_data.y,
            source_train_mask,
            model.classifier.out_features,
        )

    ckpt_path = checkpoint_path(dataset, run_idx)
    save_checkpoint(ckpt_path, model, prototypes, dataset, source_id, run_idx, args.seed + run_idx, args)

    del source_data
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path


def stage2_adapt_target(dataset, target_id, run_idx, ckpt_path, args, device):
    """阶段二：只加载目标域图、final checkpoint 和源域原型。"""
    target_data = load_graph(dataset, target_id, device)
    model, prototypes = load_checkpoint(ckpt_path, device)

    # source-free 阶段没有目标真标签，冻结源域学到的分类头，避免熵最小化把分类边界推向单类塌缩。
    for param in model.classifier.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.target_lr,
    )

    for epoch in range(args.target_epochs):
        model.train()
        optimizer.zero_grad()
        tgt_combined, tgt_homo, tgt_full, tgt_hetero = model(
            target_data.x, target_data.edge_index, target_data.x.size(0)
        )
        tgt_logits = model.classifier(tgt_combined)
        align_loss = source_free_alignment_loss(tgt_homo, tgt_full, tgt_hetero, tgt_logits, prototypes)
        entropy_loss = entropy_minimization_loss(tgt_logits)
        loss = align_loss + args.alpha * entropy_loss
        loss.backward()
        optimizer.step()

        if args.verbose and (epoch + 1) % args.print_every == 0:
            print(
                f"[{dataset} run{run_idx}] 阶段二 epoch={epoch + 1} "
                f"L_align_SF={align_loss.item():.6f} L_target={entropy_loss.item():.6f}"
            )

    if args.save_visualization_embeddings and run_idx == args.visualization_run_idx:
        model.eval()
        with torch.no_grad():
            target_features, _, _, _ = model(
                target_data.x,
                target_data.edge_index,
                target_data.x.size(0),
            )
        all_mask = eval_mask(target_data, "all")
        feat_path, labels_path = save_visualization_embeddings(
            REPO_ROOT / "visualization" / "embeddings",
            "HGDA",
            dataset,
            target_features[all_mask].detach().cpu().numpy(),
            y=target_data.y[all_mask].detach().cpu().numpy(),
            sens=target_data.sens_labels[all_mask].detach().cpu().numpy(),
        )
        print(f"[saved] {feat_path}")
        print(f"[saved] {labels_path}")

    return evaluate_target(model, target_data, args.eval_split)


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else float("nan")


def fair_metric(pred, labels, sens):
    """按 SFFGNN 口径计算 DP/EO，并输出百分制结果。"""
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    sens = np.asarray(sens).astype(int)

    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    dp = abs(safe_mean(pred[idx_s0]) - safe_mean(pred[idx_s1]))
    eo = abs(safe_mean(pred[idx_s0_y1]) - safe_mean(pred[idx_s1_y1]))
    return dp * 100.0, eo * 100.0


@torch.no_grad()
def evaluate_target(model, target_data, split):
    """目标域评估；pokec 的 y=-1 节点按 SFFGNN mask 排除。"""
    model.eval()
    mask = eval_mask(target_data, split)
    combined, _, _, _ = model(target_data.x, target_data.edge_index, target_data.x.size(0))
    logits = model.classifier(combined)
    probs_all = F.softmax(logits, dim=1)
    pred = probs_all.argmax(dim=1)[mask].detach().cpu().numpy()
    labels = target_data.y[mask].detach().cpu().numpy()
    sens = target_data.sens_labels[mask].detach().cpu().numpy()

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
    return {"Acc": acc, "AUC": auc, "DP": dp, "EO": eo}


def write_run_result(dataset, run_idx, metrics, args):
    path = run_result_path(dataset, run_idx)
    payload = {
        "method": METHOD_NAME,
        "dataset": dataset,
        "run_idx": run_idx,
        "seed": args.seed + run_idx,
        "eval_split": args.eval_split,
        "source_lr": args.source_lr,
        "target_lr": args.target_lr,
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
        f"[{METHOD_NAME}] dataset={args.dataset} run={args.run_idx} "
        f"source={source_id} target={target_id} device={device} "
        f"source_lr={args.source_lr} target_lr={args.target_lr}"
    )

    ckpt = stage1_pretrain_source(args.dataset, source_id, args.run_idx, args, device)
    metrics = stage2_adapt_target(args.dataset, target_id, args.run_idx, ckpt, args, device)
    out_path = write_run_result(args.dataset, args.run_idx, metrics, args)
    print(f"[{METHOD_NAME}] run result saved: {out_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def fmt_mean_std(values):
    arr = np.asarray(values, dtype=float)
    return f"{np.nanmean(arr):.2f}±{np.nanstd(arr):.2f}"


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


def aggregate_all(datasets, runs):
    summaries = {}
    for dataset in datasets:
        summary = aggregate_dataset(dataset, runs)
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
            f.write(f"|{dataset}|{row['Acc']}|{row['AUC']}|{row['DP']}|{row['EO']}|\n")
    return SUMMARY_FILE


def parse_gpu_ids(gpus):
    ids = []
    for item in gpus.split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return ids or [-1]


def launch_parallel(args):
    """父进程：把不同数据集和不同 run 分配到可用 GPU 上并行执行。"""
    if args.save_visualization_embeddings and not 0 <= args.visualization_run_idx < args.runs:
        raise ValueError(f"visualization_run_idx must be in [0, {args.runs - 1}]")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets
    tasks = [(dataset, run_idx) for dataset in datasets for run_idx in range(args.runs)]
    gpu_ids = parse_gpu_ids(args.gpus)
    #if not torch.cuda.is_available():
    #    gpu_ids = [-1]

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
            "--cuda",
            str(gpu_id),
            "--seed",
            str(args.seed),
            "--source_lr",
            str(args.source_lr),
            "--target_lr",
            str(args.target_lr),
            "--hidden_dim",
            str(args.hidden_dim),
            "--num_layers",
            str(args.num_layers),
            "--source_epochs",
            str(args.source_epochs),
            "--target_epochs",
            str(args.target_epochs),
            "--alpha",
            str(args.alpha),
            "--eval_split",
            args.eval_split,
            "--runs",
            str(args.runs),
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
                    f"{info['dataset']} run{info['run_idx']} failed, see {info['log_path']}"
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
    parser = argparse.ArgumentParser(description="HGDA-SF source-free runner")
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
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--save_visualization_embeddings", action="store_true")
    parser.add_argument("--visualization_run_idx", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0,1,2,4,5,6,7")
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1111)

    # 沿用原始 HGDA_MAG.py 的源域预训练学习率；目标域适配单独使用更小学习率。
    parser.add_argument("--source_lr", type=float, default=0.01)
    parser.add_argument("--target_lr", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=None, help="兼容旧参数：同时覆盖 source_lr 和 target_lr")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--source_epochs", type=int, default=50)
    parser.add_argument("--target_epochs", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--eval_split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = get_args()
    if cli_args.lr is not None:
        cli_args.source_lr = cli_args.lr
        cli_args.target_lr = cli_args.lr

    if cli_args.aggregate_only:
        out_file = aggregate_all(cli_args.datasets, cli_args.runs)
        print(f"[Aggregate] summary saved: {out_file}")
    elif cli_args.worker:
        run_single(cli_args)
    else:
        launch_parallel(cli_args)
