# coding=utf-8
"""
UDAGCN 的 source-free domain adaptation 版本。

默认入口会把 4 个数据集、每个数据集 RUN 次实验拆成独立 worker 进程，
按可用 GPU 队列并行运行。单个 worker 内部保持 UDAGCN_SF 的两阶段逻辑：
1. 源域监督预训练；
2. 删除源域数据和源图缓存后，只用目标域熵损失做 source-free 迁移。
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SFFGNN_DIR = REPO_ROOT / "SFFGNN"

RUN = 5
DEFAULT_GPUS = "0,1,2,4,5,6,7"
DEFAULT_DATASETS = ("bailA", "germanA", "pokec", "syn")

# 源/目标 id 与 SFFGNN/config.py 中注释的默认迁移方向保持一致。
DATASET_PAIRS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}

RESULT_DIR = SCRIPT_DIR / "results" / "udagcn_sf"
SUMMARY_FILE = "udagcn_sf_summary.md"


def parse_args():
    parser = argparse.ArgumentParser(description="UDAGCN_SF baseline on SFFGNN datasets")
    parser.add_argument("--mode", choices=["launch", "worker"], default="launch")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--runs", type=int, default=RUN)
    parser.add_argument("--gpus", type=str, default=DEFAULT_GPUS)
    parser.add_argument("--result_dir", type=str, default=str(RESULT_DIR))
    parser.add_argument("--base_seed", type=int, default=200)

    # UDAGCN 原始超参数，阶段一和阶段二共用。
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_prior", type=float, default=0.4)
    parser.add_argument("--encoder_dim", type=int, default=16)
    parser.add_argument("--use_udagcn", action="store_true", default=True)
    parser.add_argument("--no_udagcn", dest="use_udagcn", action="store_false")

    # worker 参数由 launcher 自动填充，手工调试单个 run 时也可直接传入。
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--source_id", type=str, default=None)
    parser.add_argument("--target_id", type=str, default=None)
    parser.add_argument("--run_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def parse_gpu_list(gpus):
    gpu_list = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_list:
        raise ValueError("GPU 列表为空；如需 CPU 调试，请显式传入 --gpus cpu 并关闭 --use_udagcn。")
    return gpu_list


def result_json_path(result_dir, dataset, run_idx):
    return Path(result_dir) / f"udagcn_sf_{dataset}_run{run_idx}.json"


def dataset_result_path(result_dir, dataset):
    return Path(result_dir) / f"udagcn_sf_{dataset}_results.txt"


def summary_path(result_dir):
    return Path(result_dir) / SUMMARY_FILE


def format_mean_std(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return "nan±nan"
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    return f"{mean:.2f}±{std:.2f}"


def load_run_results(result_dir, dataset, runs):
    rows = []
    for run_idx in range(runs):
        path = result_json_path(result_dir, dataset, run_idx)
        if not path.exists():
            return None
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def aggregate_rows(rows):
    return {
        "ACC": format_mean_std([row["target"]["acc"] for row in rows]),
        "AUC": format_mean_std([row["target"]["auc"] for row in rows]),
        "DP": format_mean_std([row["target"]["dp"] for row in rows]),
        "EO": format_mean_std([row["target"]["eo"] for row in rows]),
    }


def write_dataset_result(result_dir, dataset, rows):
    agg = aggregate_rows(rows)
    lines = [
        f"Dataset: {dataset}",
        "Metric scale: 0-100",
        "Train/eval split: all valid labeled nodes",
        "Checkpoint: final epoch",
        "",
        "|Run|ACC|AUC|DP|EO|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        target = row["target"]
        lines.append(
            "|{run}|{acc:.2f}|{auc:.2f}|{dp:.2f}|{eo:.2f}|".format(
                run=row["run_idx"],
                acc=target["acc"],
                auc=target["auc"],
                dp=target["dp"],
                eo=target["eo"],
            )
        )
    lines.extend([
        "",
        "|Dataset|ACC|AUC|DP|EO|",
        "|---|---:|---:|---:|---:|",
        f"|{dataset}|{agg['ACC']}|{agg['AUC']}|{agg['DP']}|{agg['EO']}|",
        "",
    ])
    path = dataset_result_path(result_dir, dataset)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_summary(result_dir, datasets, runs):
    lines = [
        "|Dataset|ACC|AUC|DP|EO|",
        "|---|---:|---:|---:|---:|",
    ]
    completed = []
    for dataset in datasets:
        rows = load_run_results(result_dir, dataset, runs)
        if rows is None:
            continue
        agg = aggregate_rows(rows)
        lines.append(f"|{dataset}|{agg['ACC']}|{agg['AUC']}|{agg['DP']}|{agg['EO']}|")
        completed.append(dataset)

    path = summary_path(result_dir)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, completed


def launch_all(args):
    result_dir = Path(args.result_dir)
    log_dir = result_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(args.datasets)
    for dataset in datasets:
        if dataset not in DATASET_PAIRS:
            raise ValueError(f"未知数据集 {dataset}，可选：{list(DATASET_PAIRS)}")

    gpus = parse_gpu_list(args.gpus)
    jobs = []
    # 按 run 维度交错数据集，避免启动时同一数据集占满全部 GPU。
    for run_idx in range(args.runs):
        for dataset in datasets:
            source_id, target_id = DATASET_PAIRS[dataset]
            jobs.append({
                "dataset": dataset,
                "source_id": source_id,
                "target_id": target_id,
                "run_idx": run_idx,
                "seed": args.base_seed + run_idx,
            })

    # 固定文件名复跑时先清理本次会生成的结果，避免旧 run JSON 被误汇总。
    for job in jobs:
        path = result_json_path(result_dir, job["dataset"], job["run_idx"])
        if path.exists():
            path.unlink()
    for dataset in datasets:
        path = dataset_result_path(result_dir, dataset)
        if path.exists():
            path.unlink()
    path = summary_path(result_dir)
    if path.exists():
        path.unlink()

    pending = list(jobs)
    active = []
    available_gpus = list(gpus)
    written_datasets = set()
    failed = []

    print(f"[Launcher] total jobs={len(jobs)}  gpus={gpus}  result_dir={result_dir}")

    while pending or active:
        while pending and available_gpus:
            job = pending.pop(0)
            gpu = available_gpus.pop(0)
            log_path = log_dir / f"udagcn_sf_{job['dataset']}_run{job['run_idx']}.log"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode", "worker",
                "--dataset", job["dataset"],
                "--source_id", job["source_id"],
                "--target_id", job["target_id"],
                "--run_idx", str(job["run_idx"]),
                "--seed", str(job["seed"]),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--lambda_prior", str(args.lambda_prior),
                "--encoder_dim", str(args.encoder_dim),
                "--result_dir", str(result_dir),
            ]
            if not args.use_udagcn:
                cmd.append("--no_udagcn")

            env = os.environ.copy()
            if gpu.lower() != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = gpu
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active.append({
                "proc": proc,
                "gpu": gpu,
                "job": job,
                "log_file": log_file,
                "log_path": log_path,
            })
            print(
                "[Launcher] start dataset={dataset} run={run_idx} gpu={gpu} log={log}".format(
                    dataset=job["dataset"],
                    run_idx=job["run_idx"],
                    gpu=gpu,
                    log=log_path,
                )
            )

        still_active = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                still_active.append(item)
                continue

            item["log_file"].close()
            available_gpus.append(item["gpu"])
            job = item["job"]
            if ret != 0:
                failed.append((job, item["log_path"], ret))
                print(
                    "[Launcher] failed dataset={dataset} run={run_idx} ret={ret} log={log}".format(
                        dataset=job["dataset"],
                        run_idx=job["run_idx"],
                        ret=ret,
                        log=item["log_path"],
                    )
                )
            else:
                print(
                    "[Launcher] done dataset={dataset} run={run_idx} gpu={gpu}".format(
                        dataset=job["dataset"],
                        run_idx=job["run_idx"],
                        gpu=item["gpu"],
                    )
                )

            for dataset in datasets:
                if dataset in written_datasets:
                    continue
                rows = load_run_results(result_dir, dataset, args.runs)
                if rows is not None:
                    path = write_dataset_result(result_dir, dataset, rows)
                    written_datasets.add(dataset)
                    print(f"[Launcher] dataset result saved: {path}")

        active = still_active
        if active or pending:
            time.sleep(5)

    summary, completed = write_summary(result_dir, datasets, args.runs)
    print(f"[Launcher] summary saved: {summary}")
    print(f"[Launcher] completed datasets: {completed}")

    if failed:
        print("[Launcher] failed jobs:")
        for job, log_path, ret in failed:
            print(f"  dataset={job['dataset']} run={job['run_idx']} ret={ret} log={log_path}")
        raise SystemExit(1)


def run_worker(args):
    import gc
    import random
    from types import SimpleNamespace

    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, roc_auc_score

    # 确保 dataset.py 中的相对 import 指向 SFFGNN/utils.py。
    sys.path.insert(0, str(SFFGNN_DIR))
    from dataset import get_dataset
    from utils import fair_metric

    # dual_gnn 仍使用 UDAGCN 原始实现。
    from dual_gnn.cached_gcn_conv import CachedGCNConv
    from dual_gnn.ppmi_conv import PPMIConv

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def mask_valid_binary_labels(data):
        # pokec 的标签为 {-1, 0, 1}；与 SFFGNN 对齐，监督和评估只使用 mask 中的 0/1 节点。
        valid = data.y >= 0
        data.train_mask = data.train_mask & valid
        data.val_mask = data.val_mask & valid
        data.test_mask = data.test_mask & valid
        data.y = data.y.long()
        return data

    def all_valid_mask(data):
        # 训练和评估统一使用 all split，即 train/val/test 的并集，并排除 pokec 中 y=-1 的节点。
        return (data.train_mask | data.val_mask | data.test_mask) & (data.y >= 0)

    def compute_source_prior(data, num_classes=2):
        # 阶段二只保存源域类别先验，不保留或访问源域样本，满足 source-free 设定。
        mask = all_valid_mask(data)
        labels = data.y[mask].long()
        if int(labels.numel()) == 0:
            raise ValueError(f"{args.dataset} source all split 中没有可用于统计先验的标签")
        counts = torch.bincount(labels, minlength=num_classes)[:num_classes].float()
        prior = counts / counts.sum().clamp_min(1.0)
        prior = torch.clamp(prior, min=1e-6)
        prior = prior / prior.sum()
        return prior

    class GNN(torch.nn.Module):
        def __init__(self, num_features, encoder_dim, base_model=None, type="gcn", **kwargs):
            super(GNN, self).__init__()
            if base_model is None:
                weights = [None, None]
                biases = [None, None]
            else:
                weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
                biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

            self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
            model_cls = PPMIConv if type == "ppmi" else CachedGCNConv
            self.conv_layers = nn.ModuleList([
                model_cls(num_features, 128, weight=weights[0], bias=biases[0], **kwargs),
                model_cls(128, encoder_dim, weight=weights[1], bias=biases[1], **kwargs),
            ])

        def forward(self, x, edge_index, cache_name):
            for i, conv_layer in enumerate(self.conv_layers):
                x = conv_layer(x, edge_index, cache_name)
                if i < len(self.conv_layers) - 1:
                    x = F.relu(x)
                    x = self.dropout_layers[i](x)
            return x

    class Attention(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.dense_weight = nn.Linear(in_channels, 1)
            self.dropout = nn.Dropout(0.1)

        def forward(self, inputs):
            stacked = torch.stack(inputs, dim=1)
            weights = F.softmax(self.dense_weight(stacked), dim=1)
            return torch.sum(stacked * weights, dim=1)

    class UDAGCNSF(nn.Module):
        def __init__(self, num_features, encoder_dim, num_classes=2, use_udagcn=True):
            super().__init__()
            self.use_udagcn = use_udagcn
            self.encoder = GNN(num_features, encoder_dim, type="gcn")
            if self.use_udagcn:
                self.ppmi_encoder = GNN(
                    num_features,
                    encoder_dim,
                    base_model=self.encoder,
                    type="ppmi",
                    path_len=10,
                )
                self.att_model = Attention(encoder_dim)
            self.cls_model = nn.Sequential(nn.Linear(encoder_dim, num_classes))

        def encode(self, data, cache_name, mask=None):
            gcn_output = self.encoder(data.x, data.edge_index, cache_name)
            if mask is not None:
                gcn_output = gcn_output[mask]
            if not self.use_udagcn:
                return gcn_output

            ppmi_output = self.ppmi_encoder(data.x, data.edge_index, cache_name)
            if mask is not None:
                ppmi_output = ppmi_output[mask]
            return self.att_model([gcn_output, ppmi_output])

        def predict(self, data, cache_name, mask=None):
            return self.cls_model(self.encode(data, cache_name, mask))

        def clear_cache(self):
            # 清理 CachedGCNConv/PPMIConv 的源图缓存，目标迁移阶段不再保留源图传播矩阵。
            for module in self.modules():
                if hasattr(module, "cache_dict"):
                    module.cache_dict.clear()

    def build_optimizer(model, trainable_only=False):
        if trainable_only:
            params = [param for param in model.parameters() if param.requires_grad]
        else:
            params = list(model.parameters())
        if not params:
            raise ValueError("optimizer 没有可训练参数")
        return torch.optim.Adam(params, lr=args.lr)

    def freeze_classifier(model):
        # 阶段二冻结源域预训练得到的分类边界，只更新 encoder/PPMI/attention，降低熵最小化单类坍缩风险。
        for param in model.cls_model.parameters():
            param.requires_grad_(False)

    def add_original_weight_regularization(model, loss):
        # 沿用 UDAGCN_demo.py 的权重正则实现，而不是改成新的 weight decay。
        for name, param in model.named_parameters():
            if "weight" in name:
                loss = loss + param.mean() * 3e-3
        return loss

    def pretrain_source(model, optimizer, source_data, epochs):
        model.train()
        loss_func = nn.CrossEntropyLoss().to(device)
        train_mask = all_valid_mask(source_data)
        if int(train_mask.sum().item()) == 0:
            raise ValueError(f"{args.dataset} source all split 中没有有效 0/1 标签节点")

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            source_logits = model.predict(source_data, "source")
            cls_loss = loss_func(source_logits[train_mask], source_data.y[train_mask])
            loss = add_original_weight_regularization(model, cls_loss)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0 or epoch == epochs:
                print(f"[Stage 1][{args.dataset}][run {args.run_idx}] epoch={epoch} loss={loss.item():.4f}")

    def adapt_target(model, optimizer, target_data, epochs, source_prior):
        source_prior = source_prior.to(target_data.x.device)
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            encoded_target = model.encode(target_data, "target")
            target_logits = model.cls_model(encoded_target)
            target_probs = torch.clamp(F.softmax(target_logits, dim=-1), min=1e-9, max=1.0)
            loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

            # 源域类别先验约束目标平均预测分布，排除熵最小化的单类坍缩解。
            target_mean_prob = torch.clamp(target_probs.mean(dim=0), min=1e-9, max=1.0)
            target_mean_prob = target_mean_prob / target_mean_prob.sum()
            loss_prior = torch.sum(source_prior * (torch.log(source_prior) - torch.log(target_mean_prob)))

            loss = (loss_entropy + args.lambda_prior * loss_prior) * (epoch / epochs * 0.01)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0 or epoch == epochs:
                print(
                    f"[Stage 2][{args.dataset}][run {args.run_idx}] epoch={epoch} "
                    f"entropy={loss_entropy.item():.4f} prior={loss_prior.item():.4f}"
                )

    def evaluate_target(model, target_data):
        model.eval()
        mask = all_valid_mask(target_data)
        if int(mask.sum().item()) == 0:
            raise ValueError(f"{args.dataset} target all split 中没有有效 0/1 标签节点")

        with torch.no_grad():
            logits = model.predict(target_data, "target")
            prob_matrix = F.softmax(logits, dim=-1).detach().cpu().numpy()
            probs = prob_matrix[:, 1]
            preds = logits.argmax(dim=1).detach().cpu().numpy()

        mask_np = mask.detach().cpu().numpy()
        y_true = target_data.y.detach().cpu().numpy()[mask_np]
        sens = target_data.sens_labels.detach().cpu().numpy()[mask_np].astype(int)
        prob = probs[mask_np]
        pred = preds[mask_np]

        acc = accuracy_score(y_true, pred) * 100
        auc = roc_auc_score(y_true, prob) * 100 if len(set(y_true.tolist())) == 2 else float("nan")
        dp, eo = fair_metric(pred, y_true, sens)
        mean_prob = prob_matrix[mask_np].mean(axis=0)
        pred_count = {
            str(class_id): int((pred == class_id).sum())
            for class_id in range(prob_matrix.shape[1])
        }
        return {
            "acc": float(acc),
            "auc": float(auc),
            "dp": float(dp * 100),
            "eo": float(eo * 100),
            "mean_prob": [float(value) for value in mean_prob],
            "pred_count": pred_count,
        }

    seed = args.seed if args.seed is not None else 200 + args.run_idx
    seed_everything(seed)

    if args.dataset is None:
        raise ValueError("worker 模式必须指定 --dataset")
    source_id = args.source_id or DATASET_PAIRS[args.dataset][0]
    target_id = args.target_id or DATASET_PAIRS[args.dataset][1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Worker] dataset={args.dataset} source={source_id} target={target_id} "
        f"run={args.run_idx} seed={seed} device={device}"
    )

    data_args = SimpleNamespace(dataset=args.dataset)
    source_data = mask_valid_binary_labels(get_dataset(data_args, source_id)).to(device)
    target_data = mask_valid_binary_labels(get_dataset(data_args, target_id)).to(device)

    if source_data.x.shape[1] != target_data.x.shape[1]:
        raise ValueError(
            f"{args.dataset} 源/目标特征维度不一致：{source_data.x.shape[1]} vs {target_data.x.shape[1]}"
        )

    model = UDAGCNSF(
        num_features=source_data.x.shape[1],
        encoder_dim=args.encoder_dim,
        num_classes=2,
        use_udagcn=args.use_udagcn,
    ).to(device)

    source_optimizer = build_optimizer(model)
    pretrain_source(model, source_optimizer, source_data, args.epochs)
    source_prior = compute_source_prior(source_data).to(device)

    model.clear_cache()
    del source_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 目标迁移阶段冻结分类器并重新构造优化器，沿用原始学习率，避免携带源域 Adam 动量状态。
    target_before_metrics = evaluate_target(model, target_data)
    freeze_classifier(model)
    target_optimizer = build_optimizer(model, trainable_only=True)
    adapt_target(model, target_optimizer, target_data, args.epochs, source_prior)

    # final-epoch checkpoint 策略：不使用 target 指标选最佳轮次，直接评估最终模型。
    target_metrics = evaluate_target(model, target_data)
    result = {
        "dataset": args.dataset,
        "source_id": source_id,
        "target_id": target_id,
        "run_idx": args.run_idx,
        "seed": seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_prior": args.lambda_prior,
        "encoder_dim": args.encoder_dim,
        "use_udagcn": args.use_udagcn,
        "freeze_classifier_target": True,
        "train_split": "all",
        "eval_split": "all",
        "source_prior": [float(value) for value in source_prior.detach().cpu().tolist()],
        "target_before": target_before_metrics,
        "target": target_metrics,
    }

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_json_path(result_dir, args.dataset, args.run_idx)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, out_path)
    print(f"[Worker] result saved: {out_path}")
    print(f"[Worker] target metrics: {target_metrics}")


def main():
    args = parse_args()
    if args.mode == "launch":
        launch_all(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
