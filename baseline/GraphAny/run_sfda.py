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

Usage
-----
  # Run all four pairs
  python run_sfda.py

  # Run a specific dataset pair
  python run_sfda.py --dataset pokec
  python run_sfda.py --dataset bailA
  python run_sfda.py --dataset german
  python run_sfda.py --dataset syn

Results are written to: results/graphany_sfda_results.txt
"""

import os
import sys
import random
import argparse

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

RESULTS_FILE = "results/graphany_sfda_results.txt"


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
# Training utilities
# ---------------------------------------------------------------------------

def sample_visible_nodes(train_indices, batch_nodes):
    batch_set = set(batch_nodes.tolist())
    visible   = [n for n in train_indices.tolist() if n not in batch_set]
    return torch.tensor(visible, dtype=torch.long)


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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Training GraphAny on {src_name} for {args.epochs} epochs")
    print(f"feat_channels: {feat_channels}  pred_channels: {pred_channels}")
    print("=" * 60)

    best_src_val_acc = 0.0
    best_tgt_metrics = {}

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, optimizer, criterion, src_ds,
            args.train_batch, device, cfg.n_per_label_examples,
        )

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            src_val  = evaluate(model, src_ds, "val",  args.eval_batch, device)
            tgt_test = evaluate(model, tgt_ds, "test", args.eval_batch, device)

            if src_val["acc"] > best_src_val_acc:
                best_src_val_acc = src_val["acc"]
                best_tgt_metrics = tgt_test.copy()
                ckpt = os.path.join(
                    output_dir,
                    f"graphany_{src_name}_best_srcval={best_src_val_acc:.2f}.pt",
                )
                torch.save({"state_dict": model.state_dict()}, ckpt)

            print(
                f"Epoch {epoch:4d}/{args.epochs}  loss={loss:.4f}  "
                f"src_val_acc={src_val['acc']:.2f}%  "
                f"tgt_test_acc={tgt_test['acc']:.2f}%  "
                f"tgt_test_auc={tgt_test['auc']:.2f}%"
            )

    return {
        "src": src_name,
        "tgt": tgt_name,
        "best_src_val_acc": best_src_val_acc,
        **best_tgt_metrics,
    }


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------

def write_results(results: list, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    lines.append("=" * 70)
    lines.append("GraphAny SFDA Benchmark Results")
    lines.append("=" * 70)
    lines.append(f"{'Dataset':<20} {'Src→Tgt':<22} {'ACC':>7} {'AUC':>7} {'DP':>7} {'EO':>7}")
    lines.append("-" * 70)
    for r in results:
        pair = f"{r['src']} -> {r['tgt']}"
        lines.append(
            f"{r['src'].split('_')[0]:<20} {pair:<22} "
            f"{r['acc']:>6.2f}% {r['auc']:>6.2f}% "
            f"{r['dp']:>6.2f}% {r['eo']:>6.2f}%"
        )
    lines.append("=" * 70)

    text = "\n".join(lines)
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
    parser.add_argument("--epochs",       type=int,   default=1000)
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
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
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

    all_results = []
    for ds_key, (src, tgt) in pairs_to_run:
        print(f"\n{'#'*60}")
        print(f"# Dataset pair: {src} -> {tgt}")
        print(f"{'#'*60}")
        set_seed(args.seed)
        result = run_pair(src, tgt, args)
        all_results.append(result)

        print(f"\n--- Best result for {src} -> {tgt} ---")
        print(f"  Best src_val_acc : {result['best_src_val_acc']:.2f}%")
        print(f"  Target ACC       : {result['acc']:.2f}%")
        print(f"  Target AUC       : {result['auc']:.2f}%")
        print(f"  Target delta-DP  : {result['dp']:.2f}%")
        print(f"  Target delta-EO  : {result['eo']:.2f}%")

    write_results(all_results, args.results_file)


if __name__ == "__main__":
    main()
