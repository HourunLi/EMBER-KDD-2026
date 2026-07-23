"""SOGA on the four EMBER domain pairs (5 runs, parallel, target-label selection)."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE / "codes"), str(ROOT)]

from EMBER.dataset import get_dataset  # noqa: E402
from EMBER.utils import fair_metric  # noqa: E402
from model.model import Model  # noqa: E402
from visualization.export_utils import save_visualization_embeddings  # noqa: E402


PAIRS = {
    "bailA": ("_2", "_1"),
    "germanA": ("_2", "_1"),
    "pokec": ("_z", "_n"),
    "syn": ("-2", "-1"),
}
SEEDS = [1, 3, 5, 7, 9]
SOURCE_EPOCHS = TARGET_EPOCHS = 101
SOURCE_LR = TARGET_LR = 0.01
WEIGHT_DECAY = 5e-4
STRUCT_LAMBDA = NEIGH_LAMBDA = 1.0

LOG_DIR = HERE / "logs"
RUN_DIR = HERE / "results" / "runs"
SUMMARY = HERE / "SOGA_5runs_summary.md"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logger_for(dataset, run_idx):
    logger = logging.getLogger(f"SOGA.{dataset}.{run_idx}.{os.getpid()}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def load_data(dataset, domain_id):
    return get_dataset(SimpleNamespace(dataset=dataset), domain_id)


def valid_mask(data):
    return (data.y == 0) | (data.y == 1)


def all_valid_mask(data):
    return (data.train_mask | data.val_mask | data.test_mask) & valid_mask(data)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def macro_f1(logits, labels):
    return float(
        f1_score(
            labels.detach().cpu().numpy(),
            logits.argmax(1).detach().cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )


def build_model(data, device, logger):
    args = SimpleNamespace(
        layer_unit_count_list=[data.x.shape[1], 256, 128, 2],
        gnn_model="GCN",
        head=1,
        num_label=2,
        num_target_nodes=data.num_nodes,
        metric="macro",
        struct_lambda=STRUCT_LAMBDA,
        neigh_lambda=NEIGH_LAMBDA,
        device=device,
        logger=logger,
    )
    return Model(args, logger).to(device)


def train_source(model, data, logger):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SOURCE_LR, weight_decay=WEIGHT_DECAY
    )
    val = data.val_mask & valid_mask(data)
    best_score, best_state = -1.0, None

    for epoch in range(SOURCE_EPOCHS):
        loss, _ = model.train_source(data, optimizer, None, epoch)
        model.eval()
        with torch.no_grad():
            score = macro_f1(model(data)[val], data.y[val])
        if score > best_score:
            best_score, best_state = score, cpu_state(model)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == SOURCE_EPOCHS:
            logger.info(
                "source epoch=%d/%d loss=%.6f val_macro_f1=%.6f",
                epoch + 1,
                SOURCE_EPOCHS,
                loss,
                score,
            )

    model.load_state_dict(best_state)
    logger.info("source best val_macro_f1=%.6f", best_score)


def train_target(model, data, logger):
    """Keep SOGA's loss and its target-label best-checkpoint selection."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TARGET_LR, weight_decay=WEIGHT_DECAY
    )
    select = all_valid_mask(data)  # Pokec y=-1 is excluded.
    best_score, best_state = -1.0, None

    for epoch in range(TARGET_EPOCHS):
        model.enable_target()
        logits = model(data)
        probs = F.softmax(logits, dim=-1)
        struct_nce = model.NCE_loss(
            probs,
            model.center_nodes_struct,
            model.positive_samples_struct,
            model.negative_samples_struct,
        )
        neigh_nce = model.NCE_loss(
            probs,
            model.center_nodes_neigh,
            model.positive_samples_neigh,
            model.negative_samples_neigh,
        )
        im_loss = model.ent(probs) - model.div(probs)
        loss = im_loss + STRUCT_LAMBDA * struct_nce + NEIGH_LAMBDA * neigh_nce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # The released code scores pre-update logits and saves post-update weights.
        score = macro_f1(logits[select], data.y[select])
        if score > best_score:
            best_score, best_state = score, cpu_state(model)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == TARGET_EPOCHS:
            logger.info(
                "target epoch=%d/%d loss=%.6f select_macro_f1=%.6f",
                epoch + 1,
                TARGET_EPOCHS,
                loss.item(),
                score,
            )

    model.load_state_dict(best_state)
    logger.info("target best checkpoint macro_f1=%.6f", best_score)
    return best_score


def evaluate(model, data):
    mask = all_valid_mask(data)
    model.eval()
    with torch.no_grad():
        feat = model.Extractor(data.x, data.edge_index)
        logits = model.Classifier(feat)
        prob = F.softmax(logits, dim=1)[:, 1][mask].cpu().numpy()
        pred = logits.argmax(1)[mask].cpu().numpy()
        embedding = feat[mask].cpu().numpy()

    y = data.y[mask].cpu().numpy()
    sens = data.sens_labels[mask].cpu().numpy()
    sens_ok = (sens == 0) | (sens == 1)
    dp, eo = fair_metric(pred[sens_ok], y[sens_ok], sens[sens_ok])
    return {
        "ACC": float(accuracy_score(y, pred) * 100),
        "AUC": float(roc_auc_score(y, prob) * 100),
        "DP": float(dp * 100),
        "EO": float(eo * 100),
    }, embedding, y, sens


def run_one(dataset, run_idx, seed, gpu):
    seed_all(seed)
    if gpu >= 0:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    logger = logger_for(dataset, run_idx)
    source_id, target_id = PAIRS[dataset]
    logger.info(
        "dataset=%s run=%d seed=%d source=%s target=%s device=%s",
        dataset,
        run_idx + 1,
        seed,
        source_id,
        target_id,
        device,
    )

    source = load_data(dataset, source_id).to(device)
    model = build_model(source, device, logger)
    train_source(model, source, logger)
    del source
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target = load_data(dataset, target_id)
    if target.x.shape[1] != model.args.layer_unit_count_list[0]:
        raise ValueError(f"{dataset}: source/target feature dimensions differ")
    model.args.num_target_nodes = target.num_nodes
    # Matches the effective released code: both NCE views use target edge_index.
    model.init_target(target, target)  # sample on CPU graph, tensors go to args.device
    target = target.to(device)
    best_target_f1 = train_target(model, target, logger)

    metrics, feat, y, sens = evaluate(model, target)
    logger.info("target all(valid) metrics=%s", json.dumps(metrics, sort_keys=True))
    if run_idx == 0:
        save_visualization_embeddings(
            ROOT / "visualization" / "embeddings",
            "SOGA",
            dataset,
            feat,
            y=y,
            sens=sens,
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output = RUN_DIR / f"SOGA_{dataset}_run{run_idx}.json"
    output.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "run_idx": run_idx,
                "seed": seed,
                "target_checkpoint_macro_f1": best_target_f1,
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
            json.loads((RUN_DIR / f"SOGA_{dataset}_run{i}.json").read_text())
            for i in range(len(SEEDS))
        ]
        stats = {}
        for metric in ("ACC", "AUC", "DP", "EO"):
            values = np.array([r["metrics"][metric] for r in runs])
            stats[metric] = (values.mean(), values.var(), values.std())
        rows.append((dataset, runs, stats))

    lines = [
        "# SOGA Results (5 runs)",
        "",
        "Target labels are used for best-checkpoint selection, matching the released code.",
        "Pokec `y=-1` is excluded; final evaluation is target `all(valid)`.",
        "Variance is population variance (`ddof=0`).",
        "",
        "| Dataset | Metric | Mean | Variance | Std |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, _, stats in rows:
        for metric, (mean, var, std) in stats.items():
            lines.append(f"| {dataset} | {metric} | {mean:.4f} | {var:.6f} | {std:.4f} |")

    lines += [
        "",
        "| Dataset | Run | Seed | Select Macro-F1 | ACC | AUC | DP | EO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, runs, _ in rows:
        for r in runs:
            m = r["metrics"]
            lines.append(
                f"| {dataset} | {r['run_idx'] + 1} | {r['seed']} | "
                f"{r['target_checkpoint_macro_f1']:.6f} | {m['ACC']:.4f} | "
                f"{m['AUC']:.4f} | {m['DP']:.4f} | {m['EO']:.4f} |"
            )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch(task, resources):
    gpu = resources.get()
    dataset, run_idx, seed = task
    log = LOG_DIR / f"{dataset}_run{run_idx + 1}.log"
    try:
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--dataset",
                    dataset,
                    "--run-idx",
                    str(run_idx),
                    "--seed",
                    str(seed),
                    "--gpu",
                    str(gpu),
                ],
                cwd=HERE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        resources.put(gpu)
    return dataset, run_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=PAIRS, default="bailA", help=argparse.SUPPRESS)
    parser.add_argument("--run-idx", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=SEEDS[0], help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, default=-1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_one(args.dataset, args.run_idx, args.seed, args.gpu)
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (dataset, run_idx, seed)
        for run_idx, seed in enumerate(SEEDS)
        for dataset in PAIRS
    ]
    gpu_count = torch.cuda.device_count()
    devices = list(range(gpu_count)) if gpu_count else [-1] * min(4, os.cpu_count() or 1)
    resources = Queue()
    for device in devices:
        resources.put(device)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(launch, task, resources) for task in tasks]
        for future in as_completed(futures):
            dataset, run_idx = future.result()
            print(f"done: {dataset} run{run_idx + 1}", flush=True)
    aggregate()
    print(f"summary: {SUMMARY}")


if __name__ == "__main__":
    main()
