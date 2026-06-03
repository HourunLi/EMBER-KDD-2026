"""
run_sfda.py
===========
GraphAny SFDA benchmark runner.

Trains on source dataset, evaluates zero-shot on target dataset.

Dataset pairs
-------------
  pokec   : pokec_z   -> pokec_n
  bailA   : bailA_2   -> bailA_1
  german  : german_2  -> german_1
  syn     : syn-2     -> syn-1

GPU layout (--dataset all)
--------------------------
  pokec  : 3 seeds launch first on cuda:0, cuda:1, cuda:3
  others : bailA / german / syn runs (3 seeds each) fill cuda:4/5/7
           and reuse any GPU as soon as it becomes idle

Checkpoint (aligned with SFFGNN)
--------------------------------
  Save once after the final training epoch (no best-val selection).
  Optional --use_checkpoint skips source training when a matching file exists.

Usage
-----
  python run_sfda.py                  # all four pairs
  python run_sfda.py --dataset pokec  # pokec only (3 GPUs)
  python run_sfda.py --dataset bailA
  python run_sfda.py --dataset german
  python run_sfda.py --dataset syn

Results
-------
  Per dataset (written as soon as 3 runs finish):
    results/graphany_final_{ds_key}.txt
  All four combined (written when --dataset all completes):
    results/graphany_final_all.txt
"""

import os
import sys
import random
import argparse
import queue
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

from graphany.model import GraphAny
from graphany.sfda_data import SFDAGraphDataset

# ---------------------------------------------------------------------------
# Dataset pair definitions
# ---------------------------------------------------------------------------

DATASET_PAIRS = {
    "pokec":  ("pokec_z",  "pokec_n"),
    "bailA":  ("bailA_2",  "bailA_1"),
    "german": ("german_2", "german_1"),
    "syn":    ("syn-2",    "syn-1"),
}

RESULTS_FILE = "results/graphany_final_all.txt"
N_RUNS = 3

POKEC_RUN_GPUS = ["cuda:0", "cuda:1", "cuda:3"]
OTHER_GPUS = ["cuda:4", "cuda:5"]
ALL_GPUS = POKEC_RUN_GPUS + OTHER_GPUS


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Fairness metric
# ---------------------------------------------------------------------------

def fair_metric(pred, labels, sens):
    idx_s0    = sens == 0
    idx_s1    = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity    = abs(pred[idx_s0].mean() - pred[idx_s1].mean())
    equality  = abs(pred[idx_s0_y1].mean() - pred[idx_s1_y1].mean())
    return float(parity), float(equality)


# ---------------------------------------------------------------------------
# Checkpoint helpers (SFFGNN-style: config hash + final epoch)
# ---------------------------------------------------------------------------

def _get_checkpoint_name(args, src_name: str) -> str:
    name_parts = [
        f"src={src_name}",
        f"hid={args.n_hidden}",
        f"mlay={args.n_mlp_layer}",
        f"ep={args.epochs}",
        f"lr={args.lr}",
        f"wd={args.weight_decay}",
        f"ent={args.entropy}",
        f"attn={args.attn_temp}",
        f"feat={args.feat_chn}",
        f"pred={args.pred_chn}",
        f"hops={args.n_hops}",
        f"run={args.run_idx}",
        f"seed={args.seed}",
    ]
    return "_".join(name_parts)


def _checkpoint_path(args, src_name: str) -> str:
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"{_get_checkpoint_name(args, src_name)}.pt")


def _save_checkpoint(path: str, model) -> None:
    torch.save({"state_dict": model.state_dict()}, path)
    print(f"[Checkpoint] saved → {path}")


def _load_checkpoint(path: str, model, device) -> bool:
    if not os.path.exists(path):
        return False
    print(f"[Checkpoint] loading ← {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return True


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def sample_visible_nodes(train_indices, batch_nodes):
    """Exclude batch nodes from visible reference set (same semantics as GraphAny)."""
    if batch_nodes.device != train_indices.device:
        batch_nodes = batch_nodes.to(train_indices.device)
    return train_indices[~torch.isin(train_indices, batch_nodes)]


def train_one_epoch(model, optimizer, criterion, ds, batch_size, device, n_per_label):
    model.train()
    train_idx  = ds.train_indices
    perm       = torch.randperm(len(train_idx))
    train_idx  = train_idx[perm]
    total_loss = 0.0
    n_batches  = 0

    for start in range(0, len(train_idx), batch_size):
        batch_nodes = train_idx[start: start + batch_size].to(device)

        visible = sample_visible_nodes(ds.train_indices, batch_nodes)
        if len(visible) < len(batch_nodes):
            visible = torch.cat([visible, batch_nodes[: len(batch_nodes) // 2]])
        visible = visible.to(device)

        input_logits = ds.compute_channel_logits(
            ds.features, visible, sample=True, device=device
        )
        preds, _ = model(
            {c: v[batch_nodes] for c, v in input_logits.items()},
            dist=None,
        )
        loss = criterion(preds, ds.label[batch_nodes].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, ds, split: str, batch_size: int, device: str):
    model.eval()
    eval_mask = ds.val_mask if split == "val" else ds.test_mask
    eval_idx  = eval_mask.nonzero().view(-1)

    all_preds, all_probs, all_labels, all_sens = [], [], [], []

    for start in range(0, len(eval_idx), batch_size):
        batch     = eval_idx[start: start + batch_size].to(device)
        batch_cpu = batch.cpu()

        logit_slice = {c: v[batch] for c, v in ds.unmasked_pred.items()}
        dist_slice  = ds.dist[batch].to(device)
        preds, _    = model(logit_slice, dist=dist_slice)

        all_preds.append(preds.argmax(-1).cpu())
        all_probs.append(torch.softmax(preds, dim=-1)[:, 1].cpu())
        all_labels.append(ds.label[batch].cpu())
        all_sens.append(ds.sens_labels[batch_cpu])

    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    s_true = torch.cat(all_sens).numpy().astype(int)

    acc = accuracy_score(y_true, y_pred) * 100
    try:
        auc = roc_auc_score(y_true, y_prob) * 100
    except ValueError:
        auc = float("nan")
    dp, eo = fair_metric(y_pred, y_true, s_true)

    return {"acc": acc, "auc": auc, "dp": dp * 100, "eo": eo * 100}


# ---------------------------------------------------------------------------
# Single pair runner
# ---------------------------------------------------------------------------

def run_pair(src_name: str, tgt_name: str, args) -> dict:
    device = torch.device(args.device)
    feat_channels = args.feat_chn.split("+")
    pred_channels = args.pred_chn.split("+")

    cfg = OmegaConf.create({
        "add_self_loop":         False,
        "to_bidirected":         True,
        "n_hops":                args.n_hops,
        "feat_chn":              args.feat_chn,
        "pred_chn":              args.pred_chn,
        "feat_channels":         feat_channels,
        "pred_channels":         pred_channels,
        "entropy":               args.entropy,
        "n_per_label_examples":  5,
        "seed":                  args.seed,
    })

    cache_dir  = os.path.join(args.cache_dir,  src_name.split("_")[0].replace("-", ""))
    output_dir = os.path.join(args.output_dir, src_name.split("_")[0].replace("-", ""))
    os.makedirs(cache_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Loading source dataset: {src_name}")
    src_ds = SFDAGraphDataset(
        src_name, cfg, cache_dir,
        train_batch_size=args.train_batch,
        val_test_batch_size=args.eval_batch,
        preprocess_device=device,
        seed=args.seed,
    )

    print(f"Loading target dataset: {tgt_name}")
    tgt_ds = SFDAGraphDataset(
        tgt_name, cfg, cache_dir,
        train_batch_size=args.train_batch,
        val_test_batch_size=args.eval_batch,
        preprocess_device=device,
        seed=args.seed,
    )

    src_ds.to(device)
    tgt_ds.to(device)

    model = GraphAny(
        n_hidden=args.n_hidden,
        feat_channels=feat_channels,
        pred_channels=pred_channels,
        att_temperature=args.attn_temp,
        entropy=args.entropy,
        n_mlp_layer=args.n_mlp_layer,
    ).to(device)

    ckpt_path = _checkpoint_path(args, src_name)
    skip_training = False
    if getattr(args, "use_checkpoint", False):
        skip_training = _load_checkpoint(ckpt_path, model, device)
        if skip_training:
            print(f"[{device}] Loaded checkpoint, skipping source training.")
        else:
            print(f"[{device}] No checkpoint found, training from scratch.")

    if not skip_training:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        print(f"Training GraphAny on {src_name} for {args.epochs} epochs  [{device}]")
        print(f"feat_channels: {feat_channels}  pred_channels: {pred_channels}")
        print("=" * 60)

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(
                model, optimizer, criterion, src_ds,
                args.train_batch, device, cfg.n_per_label_examples,
            )

            if epoch % args.eval_freq == 0 or epoch == args.epochs:
                src_val  = evaluate(model, src_ds, "val",  args.eval_batch, device)
                tgt_test = evaluate(model, tgt_ds, "test", args.eval_batch, device)
                print(
                    f"[{device}] Epoch {epoch:4d}/{args.epochs}  loss={loss:.4f}  "
                    f"src_val_acc={src_val['acc']:.2f}%  "
                    f"tgt_test_acc={tgt_test['acc']:.2f}%  "
                    f"tgt_test_auc={tgt_test['auc']:.2f}%"
                )

        _save_checkpoint(ckpt_path, model)

    src_val  = evaluate(model, src_ds, "val",  args.eval_batch, device)
    tgt_test = evaluate(model, tgt_ds, "test", args.eval_batch, device)
    print(
        f"[{device}] Final (epoch {args.epochs})  "
        f"src_val_acc={src_val['acc']:.2f}%  "
        f"tgt_test_acc={tgt_test['acc']:.2f}%  "
        f"tgt_test_auc={tgt_test['auc']:.2f}%"
    )

    return {
        "src": src_name,
        "tgt": tgt_name,
        "final_src_val_acc": src_val["acc"],
        **tgt_test,
    }


# ---------------------------------------------------------------------------
# Repeated runs & parallel workers
# ---------------------------------------------------------------------------

def _aggregate_runs(runs: list, src_name: str, tgt_name: str) -> dict:
    agg = {"src": src_name, "tgt": tgt_name, "n_runs": len(runs)}
    for metric in ("acc", "auc", "dp", "eo"):
        vals = np.array([r[metric] for r in runs], dtype=float)
        agg[f"{metric}_mean"] = float(vals.mean())
        agg[f"{metric}_std"] = float(vals.std())
    return agg


def _run_worker(payload):
    """Execute a single seed run."""
    ds_key, src_name, tgt_name, args_dict, device, run_idx = payload
    args = argparse.Namespace(**args_dict)
    args.device = device
    args.run_idx = run_idx
    args.seed = args_dict["seed"] + run_idx
    set_seed(args.seed)
    print(
        f"\n{'#'*60}\n"
        f"# [{device}] {ds_key} run {run_idx + 1}/{N_RUNS}  seed={args.seed}  "
        f"{src_name} -> {tgt_name}\n"
        f"{'#'*60}"
    )
    return run_pair(src_name, tgt_name, args)


def _build_run_jobs(pairs_to_run, base_seed: int):
    jobs = []
    for ds_key, (src_name, tgt_name) in pairs_to_run:
        for run_idx in range(N_RUNS):
            jobs.append((ds_key, src_name, tgt_name, run_idx, base_seed + run_idx))
    return jobs


def _make_worker_payload(job, device, args_dict):
    ds_key, src_name, tgt_name, run_idx, _seed = job
    return (ds_key, src_name, tgt_name, args_dict, device, run_idx)


def _schedule_runs(jobs, args_dict):
    """
    GPU scheduler: pokec seeds start immediately on POKEC_RUN_GPUS; other
    dataset seeds fill OTHER_GPUS first, then reuse any GPU that becomes idle.
    """
    pokec_jobs = [j for j in jobs if j[0] == "pokec"]
    other_jobs = [j for j in jobs if j[0] != "pokec"]
    results_by_ds = defaultdict(list)

    if not jobs:
        return results_by_ds

    if not torch.cuda.is_available():
        print("CUDA unavailable — running all jobs sequentially on CPU.")
        for job in jobs:
            ds_key = job[0]
            payload = _make_worker_payload(job, "cpu", args_dict)
            results_by_ds[ds_key].append(_run_worker(payload))
        return results_by_ds

    gpu_pool = queue.Queue()
    for gpu in OTHER_GPUS:
        gpu_pool.put(gpu)

    max_workers = min(len(ALL_GPUS), len(jobs))
    print(
        f"Scheduling {len(jobs)} runs: pokec priority on "
        f"{', '.join(POKEC_RUN_GPUS)}; idle GPUs from "
        f"{', '.join(OTHER_GPUS)} (+ pokec GPUs when free)."
    )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {}

        for i, job in enumerate(pokec_jobs):
            gpu = POKEC_RUN_GPUS[i]
            payload = _make_worker_payload(job, gpu, args_dict)
            fut = pool.submit(_run_worker, payload)
            future_map[fut] = (job, gpu)

        while other_jobs and not gpu_pool.empty():
            gpu = gpu_pool.get_nowait()
            job = other_jobs.pop(0)
            payload = _make_worker_payload(job, gpu, args_dict)
            fut = pool.submit(_run_worker, payload)
            future_map[fut] = (job, gpu)

        while future_map:
            done, _ = wait(future_map.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                job, gpu = future_map.pop(fut)
                ds_key = job[0]
                results_by_ds[ds_key].append(fut.result())
                gpu_pool.put(gpu)
                if other_jobs:
                    next_job = other_jobs.pop(0)
                    payload = _make_worker_payload(next_job, gpu, args_dict)
                    new_fut = pool.submit(_run_worker, payload)
                    future_map[new_fut] = (next_job, gpu)

    return results_by_ds


def run_pair_repeated(pairs_to_run, args) -> dict:
    args_dict = {k: v for k, v in vars(args).items() if k != "device"}
    jobs = _build_run_jobs(pairs_to_run, args.seed)
    results_by_ds = _schedule_runs(jobs, args_dict)

    if len(pairs_to_run) == 1:
        ds_key, (src_name, tgt_name) = pairs_to_run[0]
        runs = results_by_ds[ds_key]
        return _aggregate_runs(runs, src_name, tgt_name)

    aggregated = {}
    for ds_key, (src_name, tgt_name) in pairs_to_run:
        runs = results_by_ds[ds_key]
        aggregated[ds_key] = _aggregate_runs(runs, src_name, tgt_name)
    return aggregated


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------

def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f}±{std:.2f}%"


def dataset_results_path(results_file: str, ds_key: str) -> str:
    """Per-dataset result path: results/graphany_final_{ds_key}.txt"""
    results_dir = os.path.dirname(results_file) or "."
    return os.path.join(results_dir, f"graphany_final_{ds_key}.txt")


def _format_results_table(results: list) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append(
        f"GraphAny SFDA Benchmark Results ({N_RUNS} runs, mean±std, final-epoch checkpoint)"
    )
    lines.append("=" * 90)
    lines.append(
        f"{'Dataset':<12} {'Src→Tgt':<22} {'ACC':>14} {'AUC':>14} {'DP':>14} {'EO':>14}"
    )
    lines.append("-" * 90)
    for r in results:
        pair = f"{r['src']} -> {r['tgt']}"
        lines.append(
            f"{r['src'].split('_')[0]:<12} {pair:<22} "
            f"{_fmt_mean_std(r['acc_mean'], r['acc_std']):>14} "
            f"{_fmt_mean_std(r['auc_mean'], r['auc_std']):>14} "
            f"{_fmt_mean_std(r['dp_mean'], r['dp_std']):>14} "
            f"{_fmt_mean_std(r['eo_mean'], r['eo_std']):>14}"
        )
    lines.append("=" * 90)
    return "\n".join(lines)


def write_dataset_result(result: dict, path: str):
    """Write one dataset's aggregated result immediately after its 3 runs finish."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = _format_results_table([result])
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nDataset results written to: {path}")


def write_results(results: list, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = _format_results_table(results)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nResults written to: {path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphAny on SFDA fairness benchmark (all 4 dataset pairs)"
    )
    parser.add_argument("--dataset",      type=str,   default="all",
                        choices=["all", "pokec", "bailA", "german", "syn"],
                        help="Dataset pair to run. 'all' runs all four pairs.")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--epochs",       type=int,   default=800)
    parser.add_argument("--train_batch",  type=int,   default=128)
    parser.add_argument("--eval_batch",   type=int,   default=100000)
    parser.add_argument("--n_hidden",     type=int,   default=128)
    parser.add_argument("--n_mlp_layer",  type=int,   default=2)
    parser.add_argument("--entropy",      type=float, default=1.0)
    parser.add_argument("--attn_temp",    type=float, default=5.0)
    parser.add_argument("--feat_chn",     type=str,   default="X+L1+L2+H1+H2")
    parser.add_argument("--pred_chn",     type=str,   default="X+L1+L2")
    parser.add_argument("--n_hops",       type=int,   default=2)
    parser.add_argument("--eval_freq",    type=int,   default=50)
    parser.add_argument("--cache_dir",    type=str,   default="./data_cache")
    parser.add_argument("--output_dir",   type=str,   default="./output")
    parser.add_argument("--results_file", type=str,   default=RESULTS_FILE)
    parser.add_argument(
        "--use_checkpoint", action="store_true",
        help="load checkpoint and skip source training when available",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    pairs_to_run = (
        list(DATASET_PAIRS.items())
        if args.dataset == "all"
        else [(args.dataset, DATASET_PAIRS[args.dataset])]
    )

    if args.dataset == "all":
        aggregated = run_pair_repeated(pairs_to_run, args)
        all_results = []
        for ds_key in DATASET_PAIRS:
            if ds_key not in aggregated:
                continue
            result = aggregated[ds_key]
            write_dataset_result(
                result, dataset_results_path(args.results_file, ds_key)
            )
            all_results.append(result)
    else:
        ds_key = pairs_to_run[0][0]
        result = run_pair_repeated(pairs_to_run, args)
        write_dataset_result(
            result, dataset_results_path(args.results_file, ds_key)
        )
        all_results = [result]

    for r in all_results:
        print(f"\n--- {r['src']} -> {r['tgt']} (n={r['n_runs']}) ---")
        for metric in ("acc", "auc", "dp", "eo"):
            print(f"  {metric.upper():>3}: {_fmt_mean_std(r[f'{metric}_mean'], r[f'{metric}_std'])}")

    if args.dataset == "all":
        write_results(all_results, args.results_file)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
