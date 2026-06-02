"""
run_pokec.py
============
Entry-point to reproduce GraphAny on the Pokec domain-adaptation benchmark.

  Source (train) : pokec_z
  Target (eval)  : pokec_n

Usage
-----
  cd /home/disk2/lhr/sfda/fairDomainAdaption/GraphAny
  python run_pokec.py

The script:
  1. Loads pokec_z and pokec_n with the exact preprocessing from
     /home/disk2/lhr/sfda/code/dataset.py (via graphany/pokec_data.py).
  2. Computes propagated features (L1, L2, H1, H2) and LinearGNN channel
     logits with the same pipeline as GraphAny's GraphDataset.
  3. Trains GraphAny on pokec_z (transductive setting: train split only).
  4. Evaluates on pokec_n (inductive zero-shot transfer).
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
import torchmetrics
from omegaconf import OmegaConf
from rich.pretty import pretty_repr

# Make sure graphany package is importable
sys.path.insert(0, os.path.dirname(__file__))

from graphany.model import GraphAny
from graphany.pokec_data import PokecGraphDataset

from sklearn.metrics import accuracy_score, roc_auc_score


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


def fair_metric(pred, labels, sens):
    """
    Compute Demographic Parity (DP) and Equal Opportunity (EO).

    Args:
      pred  : np.array  binary predictions {0,1}
      labels: np.array  ground-truth labels {0,1}
      sens  : np.array  binary sensitive attribute {0,1}
    Returns:
      (parity, equality): scalar floats
    """
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity   = abs(pred[idx_s0].mean() - pred[idx_s1].mean())
    equality = abs(pred[idx_s0_y1].mean() - pred[idx_s1_y1].mean())
    return float(parity), float(equality)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphAny on Pokec (pokec_z -> pokec_n)"
    )
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--weight_decay",   type=float, default=0.02)
    parser.add_argument("--epochs",         type=int,   default=1000)
    parser.add_argument("--train_batch",    type=int,   default=128)
    parser.add_argument("--eval_batch",     type=int,   default=100000)
    parser.add_argument("--n_hidden",       type=int,   default=128)
    parser.add_argument("--n_mlp_layer",    type=int,   default=2)
    parser.add_argument("--entropy",        type=float, default=1.0)
    parser.add_argument("--attn_temp",      type=float, default=5.0)
    parser.add_argument("--feat_chn",       type=str,   default="X+L1+L2+H1+H2")
    parser.add_argument("--pred_chn",       type=str,   default="X+L1+L2")
    parser.add_argument("--n_hops",         type=int,   default=2)
    parser.add_argument("--eval_freq",      type=int,   default=50)
    parser.add_argument("--cache_dir",      type=str,   default="./data_cache")
    parser.add_argument("--output_dir",     type=str,   default="./output/pokec")
    parser.add_argument("--device",         type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def sample_visible_nodes(train_indices, batch_nodes):
    """Exclude current batch from visible reference nodes (GraphAny protocol)."""
    batch_set = set(batch_nodes.tolist())
    visible   = [n for n in train_indices.tolist() if n not in batch_set]
    return torch.tensor(visible, dtype=torch.long)


def train_one_epoch(model, optimizer, criterion, ds, batch_size, device,
                    n_per_label):
    model.train()
    train_idx = ds.train_indices
    # Shuffle
    perm       = torch.randperm(len(train_idx))
    train_idx  = train_idx[perm]
    total_loss = 0.0
    n_batches  = 0

    for start in range(0, len(train_idx), batch_size):
        batch_nodes = train_idx[start: start + batch_size].to(device)

        # Visible nodes = train set minus current batch (GraphAny training protocol)
        visible = sample_visible_nodes(ds.train_indices, batch_nodes)
        if len(visible) < len(batch_nodes):
            # Dataset too small: add first half of batch as visible
            visible = torch.cat([visible, batch_nodes[: len(batch_nodes) // 2]])
        visible = visible.to(device)

        # Compute channel logits from visible nodes (bootstrapped LinearGNN)
        input_logits = ds.compute_channel_logits(
            ds.features, visible, sample=True, device=device
        )

        preds, _ = model(
            {c: v[batch_nodes] for c, v in input_logits.items()},
            dist=None,   # recompute dist during training
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
    """Evaluate on val or test split of *ds* using pre-computed unmasked_pred."""
    model.eval()
    if split == "val":
        eval_mask = ds.val_mask
    else:
        eval_mask = ds.test_mask

    eval_idx  = eval_mask.nonzero().view(-1)
    all_preds  = []
    all_probs  = []
    all_labels = []
    all_sens   = []

    for start in range(0, len(eval_idx), batch_size):
        batch = eval_idx[start: start + batch_size].to(device)
        batch_cpu = batch.cpu()
        # Use pre-computed unmasked predictions (full-graph) for evaluation
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

    # 计算指标
    acc = accuracy_score(y_true, y_pred) * 100
    try:
        auc = roc_auc_score(y_true, y_prob) * 100
    except ValueError:
        auc = float("nan")
        
    dp, eo = fair_metric(y_pred, y_true, s_true)
    
    # 统一转换为百分制
    return {"acc": acc, "auc": auc, "dp": dp * 100, "eo": eo * 100}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir,  exist_ok=True)

    feat_channels = args.feat_chn.split("+")
    pred_channels = args.pred_chn.split("+")

    # Build a minimal OmegaConf config so PokecGraphDataset behaves like
    # GraphDataset (it reads the same keys).
    cfg = OmegaConf.create({
        "add_self_loop":  False,
        "to_bidirected":  True,
        "n_hops":         args.n_hops,
        "feat_chn":       args.feat_chn,
        "pred_chn":       args.pred_chn,
        "feat_channels":  feat_channels,
        "pred_channels":  pred_channels,
        "entropy":        args.entropy,
        "n_per_label_examples": 5,
        "seed":           args.seed,
    })

    preprocess_device = device

    print("=" * 60)
    print("Loading source dataset: pokec_z")
    src_ds = PokecGraphDataset(
        "pokec_z", cfg, args.cache_dir,
        train_batch_size=args.train_batch,
        val_test_batch_size=args.eval_batch,
        preprocess_device=preprocess_device,
        seed=args.seed,
    )

    print("=" * 60)
    print("Loading target dataset: pokec_n")
    tgt_ds = PokecGraphDataset(
        "pokec_n", cfg, args.cache_dir,
        train_batch_size=args.train_batch,
        val_test_batch_size=args.eval_batch,
        preprocess_device=preprocess_device,
        seed=args.seed,
    )

    # Move datasets to device
    src_ds.to(device)
    tgt_ds.to(device)

    # ---- Build GraphAny model -------------------------------------------
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

    print("=" * 60)
    print(f"Training GraphAny on pokec_z for {args.epochs} epochs")
    print(f"feat_channels : {feat_channels}")
    print(f"pred_channels : {pred_channels}")
    print("=" * 60)

    best_src_val_acc  = 0.0
    best_tgt_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, optimizer, criterion, src_ds,
            args.train_batch, device, cfg.n_per_label_examples
        )

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            src_val_metrics  = evaluate(model, src_ds, "val",  args.eval_batch, device)
            src_test_metrics = evaluate(model, src_ds, "test", args.eval_batch, device)
            tgt_val_metrics  = evaluate(model, tgt_ds, "val",  args.eval_batch, device)
            tgt_test_metrics = evaluate(model, tgt_ds, "test", args.eval_batch, device)

            # 依旧使用 src_val 的 acc 作为最佳模型保存的标准
            if src_val_metrics["acc"] > best_src_val_acc:
                best_src_val_acc  = src_val_metrics["acc"]
                best_tgt_metrics  = tgt_test_metrics
                ckpt_path = os.path.join(
                    args.output_dir,
                    f"graphany_pokec_best_srcval={best_src_val_acc:.2f}.pt"
                )
                torch.save(
                    {"state_dict": model.state_dict(),
                     "graph_any_config": {
                         "n_hidden":       args.n_hidden,
                         "feat_channels":  feat_channels,
                         "pred_channels":  pred_channels,
                         "att_temperature":args.attn_temp,
                         "entropy":        args.entropy,
                         "n_mlp_layer":    args.n_mlp_layer,
                     }},
                    ckpt_path,
                )

            print(
                f"Epoch {epoch:4d}/{args.epochs}  loss={loss:.4f}  "
                f"src_val_acc={src_val_metrics['acc']:.2f}%  "
                f"tgt_test_acc={tgt_test_metrics['acc']:.2f}%  "
                f"tgt_test_auc={tgt_test_metrics['auc']:.2f}%"
            )

    print("=" * 60)
    print("Training complete.")
    print("=== Target (pokec_n) Best Results (Selected by Source Val) ===")
    print(f"Best src_val_acc  : {best_src_val_acc:.2f}%")
    print(f"Target ACC        : {best_tgt_metrics['acc']:.2f}%")
    print(f"Target ROC-AUC    : {best_tgt_metrics['auc']:.2f}%")
    print(f"Target delta-DP   : {best_tgt_metrics['dp']:.2f}%")
    print(f"Target delta-EO   : {best_tgt_metrics['eo']:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
