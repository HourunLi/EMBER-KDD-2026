from tqdm import tqdm
import numpy as np
import os
import sys
from models import *
from config import mprint
from utils import *
from learn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

try:
    from .adaptation import (
        load_adaptation_state,
        predict_target_proba,
        run_target_adaptation,
        save_adaptation_state,
    )
except ImportError:
    from adaptation import (
        load_adaptation_state,
        predict_target_proba,
        run_target_adaptation,
        save_adaptation_state,
    )

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from visualization.export_utils import save_visualization_embeddings

# 1) 源域上的 FairGNN 训练；
# 2) 从源模型中提取“可迁移知识”（prototype / residual / prior）；
# 3) 在不访问源域原始数据的前提下，对目标域做 SFDA 适配；
# 4) 对适配前后结果做评估。
#
# “先在 source 学到一个兼顾分类与公平性的表示空间，再把这个空间中的统计知识
#  迁移到 target，用 prototype 对齐 + 伪标签自训练的方式继续适配。”


# ── Fairness losses ────────────────────────────────────────────────────────────

def mmd_loss(h, s, bandwidth=1.0, chunk_size=1024):
    """
    MMD (Maximum Mean Discrepancy) between embeddings of sensitive group 0 and 1.
    Measures distributional fairness in the embedding space.

    MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
    where k is an RBF kernel.

    This implementation is chunked to avoid materialising a full [n, m, D]
    tensor, which can easily OOM on larger datasets.

    Args:
      h:          [N, D]  node embeddings
      s:          [N]     binary sensitive attribute {0, 1}
      bandwidth:  float   RBF kernel bandwidth σ
      chunk_size: int     block size for pairwise kernel computation
    Returns:
      scalar MMD² loss (0.0 if either group is empty)
    """
    h0 = h[s == 0]   # [N0, D]
    h1 = h[s == 1]   # [N1, D]
    if h0.shape[0] == 0 or h1.shape[0] == 0:
        return h.new_zeros(1).squeeze()

    def rbf_mean_chunked(x, y):
        # 精确计算 mean_{i,j} exp(-||x_i-y_j||^2 / (2σ^2))，
        # 但按块处理，避免构造完整的 [n, m, D] 张量。
        total = x.new_zeros(())
        count = 0
        denom = 2 * (bandwidth ** 2)

        for i in range(0, x.shape[0], chunk_size):
            xb = x[i:i + chunk_size]                                # [bx, D]
            x_sq = xb.pow(2).sum(dim=1, keepdim=True)               # [bx, 1]

            for j in range(0, y.shape[0], chunk_size):
                yb = y[j:j + chunk_size]                            # [by, D]
                y_sq = yb.pow(2).sum(dim=1).unsqueeze(0)            # [1, by]
                dist2 = (x_sq + y_sq - 2.0 * xb @ yb.t()).clamp_min(0.0)
                total = total + torch.exp(-dist2 / denom).sum()
                count += xb.shape[0] * yb.shape[0]

        return total / max(count, 1)

    return rbf_mean_chunked(h0, h0) + rbf_mean_chunked(h1, h1) - 2.0 * rbf_mean_chunked(h0, h1)



def _init_fair_gnn(args):
    """
    Initialise FairGNN with Adam optimiser and a two-phase LR schedule:
      Phase 1 — Linear warm-up over the first `warmup_epochs` epochs
                 (lr ramps from lr/10 up to args.lr).
      Phase 2 — Cosine annealing for the remaining epochs
                 (lr decays smoothly from args.lr down to args.lr/100).
    """
    model     = FairGNN(args, encoder_type=args.inter_encoder).to(args.device)
    # Optimizer starts at peak lr; LinearLR will scale it via start_factor
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr2_reg)

    warmup_epochs  = max(1, int(args.train_epochs * 0.2))   # 10% of total epochs
    cosine_epochs  = args.train_epochs - warmup_epochs

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,          # begins at lr*0.1, ramps up to lr*1.0
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=args.lr / 100,     # floor at 1% of peak lr
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return model, optimizer, scheduler


# ── Source knowledge extraction ──────────────────────────────────────────────

def extract_source_knowledge(args, data, model):
    """
    Export source knowledge from a trained model.  Nothing here touches raw
    data after this call — everything downstream uses only the returned dict.

    -------------------
    p_y   : {0: [D], 1: [D]}
        Class prototype — mean of L2-normalised embeddings for each class y,
        computed over TRAINING nodes only to stay consistent with the source
        supervision used to train the model.

            p_y^S = (1 / |D_y^{train}|) * sum_{i: i in train, y_i=y}  Norm(h_i)

    r_ys  : {(y,s): [D]}
        Sensitive residual — difference between the joint (y,s) prototype
        and the class prototype:

            p_{y,s}^S = (1 / |D_{y,s}^{train}|) * sum_{i: i in train, y_i=y, s_i=s}  Norm(h_i)
            r_{y,s}^S = p_{y,s}^S - p_y^S

    pi_ys : {(y,s): float}
        Empirical joint prior P(y,s) estimated from training nodes only.

    Also saves model weights (backbone + cls_head) so the encoder can be
    reloaded without the original source data.

    Returns
    -------
    dict with keys: 'p_y', 'r_ys', 'pi_ys', 'model_state'
    """
    # 不把源数据本身带到目标域，而是把源域中已经学到的统计结构导出来：
    #
    # - p_y:   类别原型，描述“某个类别大概在表示空间的什么位置”
    # - r_ys:  敏感属性残差，描述“同一类别里不同敏感组的偏移”
    # - pi_ys: 联合先验，描述 (y,s) 组合在源域里出现的频率
    #
    # 这样后续 target 适配阶段就不需要再访问 source 原始样本，
    # 满足 source-free domain adaptation 的设定。
    model.eval()
    with torch.no_grad():
        emb, _ = model(data.x, data.edge_index)          # [N, D]
        emb_n  = F.normalize(emb, p=2, dim=1)            # L2-normalised

    y  = data.y.cpu()
    s  = data.sens_labels.cpu()
    tm = data.train_mask.cpu()

    # source model 的监督来自 train_mask，因此知识抽取也只统计训练节点，
    # 避免把 source 保留集标签再次注入到可迁移知识中。
    emb_stats = emb_n[tm]
    y_stats   = y[tm]
    s_stats   = s[tm]

    p_y = {}
    for yv in [0, 1]:
        mask = (y_stats == yv)
        if mask.sum() == 0:
            p_y[yv] = torch.zeros(emb_n.shape[1], device='cpu')
        else:
            p_y[yv] = emb_stats[mask].mean(dim=0).cpu()

    # joint prototypes p_ys and sensitive residuals r_ys (training nodes only)
    # r_ys：在固定类别 y 下，敏感组 s 会让表示产生怎样的偏移。
    # 后面在 target 域中，模型会在 class prototype 基础上再加这个 residual，形成更细粒度的 joint prototype。
    r_ys = {}
    for yv in [0, 1]:
        for sv in [0, 1]:
            mask = (y_stats == yv) & (s_stats == sv)
            if mask.sum() == 0:
                p_ys = p_y[yv].clone()
            else:
                p_ys = emb_stats[mask].mean(dim=0).cpu()
            r_ys[(yv, sv)] = p_ys - p_y[yv]

    # empirical joint prior pi_ys (training nodes only)
    n_train = tm.sum().item()
    pi_ys = {}
    for yv in [0, 1]:
        for sv in [0, 1]:
            mask = tm & (y == yv) & (s == sv)
            pi_ys[(yv, sv)] = mask.sum().item() / max(n_train, 1)

    return {
        'p_y':        p_y,
        'r_ys':       r_ys,
        'pi_ys':      pi_ys,
        'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
    }


def save_source_knowledge(knowledge, path):
    """Persist the source knowledge dict to disk."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(knowledge, path)
    print(f'[Source knowledge] saved → {path}')


def _get_checkpoint_name(args, run_idx):
    """
    Generate a unique checkpoint name based on model/training config.
    This ensures different configs don't overwrite each other.
    """
    name_parts = [
        f"ds={args.dataset}",
        f"src={args.inid}",
        f"enc={args.inter_encoder}",
        f"hid={args.hidden_dim}",
        f"lay={args.n_layers}",
        f"ep={args.train_epochs}",
        f"lr={args.lr}",
        f"drop={args.dropout}",
        f"lfair={args.lambda_fair}",
        f"run={run_idx}",
        f"seed={args.seed}",
    ]
    return "_".join(name_parts)


def _checkpoint_path(args, run_idx):
    """Return the full path for a checkpoint."""
    import os
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    name = _get_checkpoint_name(args, run_idx)
    return os.path.join(ckpt_dir, f"{name}.pt")


def _save_checkpoint(args, run_idx, model, knowledge):
    """Save model state and source knowledge to a checkpoint file."""
    path = _checkpoint_path(args, run_idx)
    ckpt = {
        'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'knowledge': knowledge,
    }
    torch.save(ckpt, path)
    print(f'[Checkpoint] saved → {path}')


def _load_checkpoint(args, run_idx):
    """
    Load checkpoint if it exists.
    Returns (model, knowledge) if found, else (None, None).
    """
    import os
    path = _checkpoint_path(args, run_idx)
    if not os.path.exists(path):
        return None, None
    print(f'[Checkpoint] loading ← {path}')
    ckpt = torch.load(path, map_location=args.device)
    # Reconstruct model
    model = FairGNN(args, encoder_type=args.inter_encoder).to(args.device)
    model.load_state_dict({k: v.to(args.device) for k, v in ckpt['model_state'].items()})
    return model, ckpt['knowledge']


# SFDA helpers
class ResidualScaler(nn.Module):
    """
    Learnable per-dimension scale vector: Δ_{y,s}(h) = γ_{y,s} ⊙ h
    init_weight > 0 gives minor-attribute groups a head-start so their
    residuals are updated more aggressively at the start of adaptation.
    """
    def __init__(self, dim: int, device, init_weight: float = 0.0):
        super().__init__()
        self.gamma = nn.Parameter(
            torch.full((dim,), init_weight, device=device)
        )

    def forward(self, h):
        return self.gamma * h   # [N, D] or [D]


def _init_residual_scalers(dim, dev, pi_ys, keys):
    """Build safe zero-initialised scalers for legacy cold-state inference."""
    del pi_ys
    return {k: ResidualScaler(dim, dev, init_weight=0.0) for k in keys}


def _build_sfda_state(args, model, target_data, p_y, r_ys, pi_ys, tau_adjust, temp, keys, scalers=None):
    """
    构造用于 SFDA 推断/评估的状态。
    若提供 scalers，则使用与适配阶段一致的原型构造方式生成 final_protos。
    """
    state = {
        'p_y':        {yv: p_y[yv].cpu() for yv in [0, 1]},
        'r_ys':       {k:  v.cpu()       for k, v in r_ys.items()},
        'pi_ys':      pi_ys,
        'tau_adjust': tau_adjust,
        'proto_temp': temp,
    }

    with torch.no_grad():
        if scalers is None:
            final_protos = {}
            for (yv, sv) in keys:
                raw = p_y[yv] + r_ys[(yv, sv)]
                final_protos[(yv, sv)] = F.normalize(raw, p=2, dim=0)
        else:
            emb_raw, _ = model(target_data.x, target_data.edge_index)
            h = F.normalize(emb_raw, p=2, dim=1)

            coarse_mean = {yv: h.mean(0) for yv in [0, 1]}
            protos_coarse = _build_protos(p_y, r_ys, scalers, coarse_mean, keys)
            _, q_y_coarse = _compute_posterior(
                h, protos_coarse, pi_ys, args.lambda_pi, keys, tau_adjust, temp
            )

            h_mean_y = {}
            for yv in [0, 1]:
                w = q_y_coarse[:, yv]
                h_mean_y[yv] = (h * w.unsqueeze(1)).sum(0) / w.sum().clamp(min=1e-8)

            final_protos = _build_protos(p_y, r_ys, scalers, h_mean_y, keys)

    state['final_protos'] = {k: v.cpu() for k, v in final_protos.items()}
    return state


def _build_source_free_eval_state(args, model, target_data, knowledge):
    """
    构造适配前的冷启动评估状态，和 SFDA 第一次前向的口径保持一致。
    """
    dev = args.device
    tau_adjust = getattr(args, 'tau_adjust', 1.0)
    temp       = getattr(args, 'proto_temp', 0.5)
    keys       = [(0, 0), (0, 1), (1, 0), (1, 1)]

    p_y = {yv: knowledge['p_y'][yv].clone().to(dev) for yv in [0, 1]}
    r_ys = {k: v.clone().to(dev) for k, v in knowledge['r_ys'].items()}
    pi_ys = {k: float(v) for k, v in knowledge['pi_ys'].items()}

    dim = knowledge['p_y'][0].shape[0]
    scalers = _init_residual_scalers(dim, dev, pi_ys, keys)

    return _build_sfda_state(
        args, model, target_data, p_y, r_ys, pi_ys, tau_adjust, temp, keys, scalers=scalers
    )


# SFDA Target Adaptation
def _adapt_target_shared(args, target_data, knowledge):
    """Load the frozen source model and run FairMAC target adaptation."""
    dev = args.device
    model = FairGNN(args, encoder_type=args.inter_encoder).to(dev)
    model.load_state_dict({k: v.to(dev) for k, v in knowledge['model_state'].items()})
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    target_data = target_data.to(dev)
    state = run_target_adaptation(args, model, target_data, knowledge)
    return model, state


def adapt_target(args, target_data, knowledge):
    """Run FairMAC target adaptation."""
    return _adapt_target_shared(args, target_data, knowledge)




def _build_protos(p_y, r_ys, scalers, h_mean_y, keys):
    """
    Build 4 joint prototypes with learned residual correction.
      p̃_{y,s} = Norm( p_y^T + r_{y,s}^T + γ_{y,s} ⊙ h_mean_y )
    where h_mean_y is the class-wise mean of target embeddings for class y.
    Returned tensors carry grad through γ.
    """
    protos = {}
    for (yv, sv) in keys:
        delta = scalers[(yv, sv)](h_mean_y[yv])               # [D], has grad
        raw   = p_y[yv] + r_ys[(yv, sv)] + delta
        protos[(yv, sv)] = F.normalize(raw, p=2, dim=0)
    return protos


def _compute_posterior(h, protos, pi_ys, lambda_pi, keys, tau_adjust=1.0, temp=0.1):
    """
    BCA-style joint posterior with count-based Bayesian logit adjustment.

      z_i(y,s) = cosine_sim(h_i, p̃_{y,s}) / temp + τ · λ_π · log π_{y,s}
      q_i(y,s) = softmax_over_(y,s)( z_i )
      q_i(y)   = Σ_s q_i(y,s)

    temp: softmax temperature applied ONLY to the likelihood (cosine sim).
          Lower temp → sharper distribution based on similarity.
          The prior term is NOT scaled by temp to avoid prior domination.

    Returns q_ys [N,4], q_y [N,2].
    """
    scores = []
    for (yv, sv) in keys:
        sim = (h * protos[(yv, sv)].unsqueeze(0)).sum(1)       # [N]
        pi  = max(pi_ys[(yv, sv)], 1e-8)
        # temp only scales the likelihood (sim), not the prior
        scores.append(sim / temp + tau_adjust * lambda_pi * float(np.log(pi)))
    scores = torch.stack(scores, dim=1)                        # [N,4]
    q_ys   = torch.softmax(scores, dim=1)
    q_y    = torch.stack([
        q_ys[:, 0] + q_ys[:, 1],
        q_ys[:, 2] + q_ys[:, 3],
    ], dim=1)                                                  # [N,2]
    return q_ys, q_y



# Training + Adaptation

def train_and_adapt(args, source_data, target_data):
    """
    For each run:
      1. Train FairGNN on source domain.
      2. Extract source knowledge.
      3. SFDA adapt to target domain.
      4. Evaluate source / target-before / target-after.
    """
    # Source metrics
    src_acc      = np.zeros([args.runs, 1])
    src_auc_roc  = np.zeros([args.runs, 1])
    src_parity   = np.zeros([args.runs, 1])
    src_equality = np.zeros([args.runs, 1])
    # Target before adaptation
    tgt_acc      = np.zeros([args.runs, 1])
    tgt_auc_roc  = np.zeros([args.runs, 1])
    tgt_parity   = np.zeros([args.runs, 1])
    tgt_equality = np.zeros([args.runs, 1])
    # Target after adaptation
    ada_acc      = np.zeros([args.runs, 1])
    ada_auc_roc  = np.zeros([args.runs, 1])
    ada_parity   = np.zeros([args.runs, 1])
    ada_equality = np.zeros([args.runs, 1])

    source_data = source_data.to(args.device)
    target_data = target_data.to(args.device)

    cls_labels  = source_data.y.float().to(args.device)
    sens_labels = source_data.sens_labels.to(args.device)
    train_mask  = source_data.train_mask

    criterion = nn.BCEWithLogitsLoss()

    for run_idx in tqdm(range(args.runs), unit='run'):

        # ── Try to load existing checkpoint ────────────────────────────────
        model, knowledge = (None, None)
        if getattr(args, 'use_checkpoint', True):
            model, knowledge = _load_checkpoint(args, run_idx)
        skip_training = (model is not None and knowledge is not None)

        if skip_training:
            print(f"[Run {run_idx}] Loaded checkpoint, skipping training.")
        else:
            if getattr(args, 'use_checkpoint', True):
                print(f"[Run {run_idx}] No checkpoint found, training from scratch.")
            else:
                print(f"[Run {run_idx}] Checkpoint loading disabled, training from scratch.")
            # ── Phase 1: Source training ───────────────────────────────────────
            model, optimizer, scheduler = _init_fair_gnn(args)

            meta_lr_src = getattr(args, 'meta_lr_src', 0.01)

            for epoch in tqdm(range(args.train_epochs), desc=f'Run {run_idx} [train]', leave=False):
                model.train()

                _, cls_logit = model(source_data.x, source_data.edge_index)

                # ── Classification loss (inner task) ────────────────────────────
                L_cls = criterion(cls_logit[train_mask].view(-1), cls_labels[train_mask])

                # ── Meta-learning: cls as inner task, MMD fairness as outer task ─
                #
                # Inner step: θ' = θ - meta_lr · ∇_θ L_cls
                # Outer loss: L_mmd(θ') — MMD on embeddings from updated params
                #
                # This ensures cls updates stay compatible with fairness alignment:
                # the model learns to classify in a way that doesn't hurt MMD.
                backbone_params = list(model.backbone.parameters())
                grad_cls = torch.autograd.grad(
                    L_cls, backbone_params, create_graph=True, retain_graph=True
                )
                # Virtual inner step (functional, no in-place update)
                params_prime = [
                    p - meta_lr_src * g
                    for p, g in zip(backbone_params, grad_cls)
                ]

                # Forward with θ' using functional_call
                param_dict = dict(zip(
                    [n for n, _ in model.backbone.named_parameters()],
                    params_prime
                ))
                emb_prime = functional_call(
                    model.backbone, param_dict,
                    (source_data.x, source_data.edge_index)
                )
                emb_prime_train = emb_prime[train_mask]

            # Outer fairness loss: MMD between sensitive groups in θ' embedding space
                L_mmd = mmd_loss(
                    emb_prime_train,
                    sens_labels[train_mask].long(),
                    chunk_size=getattr(args, 'mmd_chunk_size', 1024),
                )

                loss = L_cls + args.lambda_fair * L_mmd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (epoch + 1) % 50 == 0:
                    print(f"[Run {run_idx} | Epoch {epoch+1:03d}] "
                          f"L_cls={L_cls.item():.4f}  "
                          f"L_mmd={L_mmd.item():.4f}  "
                          f"Total={loss.item():.4f}")

            # ── Extract and save source knowledge ──────────────────────────────
            knowledge = extract_source_knowledge(args, source_data, model)
            _save_checkpoint(args, run_idx, model, knowledge)

        # ── Evaluate on source ─────────────────────────────────────────────
        accs, auc_rocs, tmp_parity, tmp_equality = evaluate_per_class(
            args, source_data, model
        )
        print(f"[Run {run_idx}] Source | "
              f"Acc={accs['all']:.2f}  AUC={auc_rocs['all']:.2f}  "
              f"DP={tmp_parity['all']:.2f}  EO={tmp_equality['all']:.2f}")
        src_acc[run_idx]      = accs['all']
        src_auc_roc[run_idx]  = auc_rocs['all']
        src_parity[run_idx]   = tmp_parity['all']
        src_equality[run_idx] = tmp_equality['all']

        # ── Evaluate on target (before adaptation) ─────────────────────────
        cold_state = _build_source_free_eval_state(args, model, target_data, knowledge)
        t_accs, t_auc_rocs, t_parity, t_equality = evaluate_after(
            args,
            target_data,
            model,
            cold_state,
        )
        print(f"[Run {run_idx}] Target (before adapt) | "
              f"Acc={t_accs['all']:.2f}  AUC={t_auc_rocs['all']:.2f}  "
              f"DP={t_parity['all']:.2f}  EO={t_equality['all']:.2f}")
        tgt_acc[run_idx]      = t_accs['all']
        tgt_auc_roc[run_idx]  = t_auc_rocs['all']
        tgt_parity[run_idx]   = t_parity['all']
        tgt_equality[run_idx] = t_equality['all']

        # ── Phase 2: SFDA adapt ────────────────────────────────────────────
        # knowledge was either loaded from checkpoint or extracted above
        if getattr(args, 'target_seed', None) is not None:
            seed_everything(args.target_seed)
        adapted_model, state = adapt_target(args, target_data, knowledge)

        # ── Evaluate on target (after adaptation) ──────────────────────────
        a_accs, a_aucs, a_par, a_eq = evaluate_after(
            args,
            target_data,
            adapted_model,
            state,
            save_visualization=getattr(args, 'save_visualization_embeddings', False),
        )
        print(f"[Run {run_idx}] Target (after adapt) | "
              f"Acc={a_accs['all']:.2f}  AUC={a_aucs['all']:.2f}  "
              f"DP={a_par['all']:.2f}  EO={a_eq['all']:.2f}")
        ada_acc[run_idx]      = a_accs['all']
        ada_auc_roc[run_idx]  = a_aucs['all']
        ada_parity[run_idx]   = a_par['all']
        ada_equality[run_idx] = a_eq['all']

    return (src_acc, src_auc_roc, src_parity, src_equality,
            tgt_acc, tgt_auc_roc, tgt_parity, tgt_equality,
            ada_acc, ada_auc_roc, ada_parity, ada_equality)


def evaluate_after(args, data, encoder, state, save_visualization=False):
    """
    Evaluate on target (after adaptation)
    Returns:
      accs, auc_rocs, paritys, equalitys  — each a dict keyed by split name
    """
    sens_labels = data.sens_labels
    test_labels = data.y[data.test_mask]

    t_idx_s0 = sens_labels[data.test_mask] == 0
    t_idx_s1 = sens_labels[data.test_mask] == 1
    t_idx_s0_y1 = torch.logical_and(t_idx_s0, test_labels == 1)
    t_idx_s1_y1 = torch.logical_and(t_idx_s1, test_labels == 1)
    t_idx_s0_y0 = torch.logical_and(t_idx_s0, test_labels == 0)
    t_idx_s1_y0 = torch.logical_and(t_idx_s1, test_labels == 0)

    accs, auc_rocs, f1s, paritys, equalitys = {}, {}, {}, {}, {}

    encoder.eval()
    with torch.no_grad():
        if int(state.get('state_version', 0)) >= 2:
            feat, q_y = predict_target_proba(args, data, encoder, state)
        else:
            emb_raw, _ = encoder(data.x, data.edge_index)
            feat = F.normalize(emb_raw, p=2, dim=1)
            p_y = {k: v.to(args.device) for k, v in state['p_y'].items()}
            r_ys = {k: v.to(args.device) for k, v in state['r_ys'].items()}
            pi_ys = state['pi_ys']
            tau_adjust = state.get('tau_adjust', 1.0)
            temp = state.get('proto_temp', 1)
            keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
            if 'final_protos' in state:
                final_protos = {
                    k: v.to(args.device) for k, v in state['final_protos'].items()
                }
            else:
                final_protos = {}
                for (yv, sv) in keys:
                    raw = p_y[yv] + r_ys[(yv, sv)]
                    final_protos[(yv, sv)] = F.normalize(
                        raw, p=2, dim=0, eps=1e-12
                    )
            _, q_y = _compute_posterior(
                feat,
                final_protos,
                pi_ys,
                args.lambda_pi,
                keys,
                tau_adjust,
                temp,
            )
        
        # extract positive probability
        probs = q_y[:, 1].cpu().numpy()

        y_all   = data.y.cpu().numpy()
        sens_all = data.sens_labels.cpu().numpy()
        all_mask = data.train_mask | data.val_mask | data.test_mask

        splits = {
            'all':   all_mask.cpu().numpy(),
            'train': data.train_mask.cpu().numpy(),
            'val':   data.val_mask.cpu().numpy(),
            'test':  data.test_mask.cpu().numpy(),
        }

        # Save embeddings only when explicitly requested.  Random-tune workers
        # otherwise race while overwriting the same dataset-level NPZ files.
        labels = torch.full((test_labels.shape[0],), -1, dtype=torch.int64)
        labels[t_idx_s0_y1] = 0
        labels[t_idx_s1_y1] = 1
        labels[t_idx_s0_y0] = 2
        labels[t_idx_s1_y0] = 3
        export_embeddings = not getattr(args, 'disable_embedding_export', False)
        if export_embeddings:
            np.savez(f"{args.dataset}_feat.npz",
                     representations=feat[data.test_mask].cpu().numpy())
            np.savez(f"{args.dataset}_labels.npz",
                     labels=labels.cpu().numpy())
        if (export_embeddings and save_visualization
                and getattr(args, 'save_visualization_embeddings', False)):
            embeddings_root = os.path.join(PROJECT_ROOT, 'visualization', 'embeddings')
            save_visualization_embeddings(
                embeddings_root,
                'SFFGNN',
                args.dataset,
                feat[data.test_mask].cpu().numpy(),
                labels=labels.cpu().numpy(),
            )

        result = {}
        for split_name, mask in splits.items():
            y_true = y_all[mask]
            sens   = sens_all[mask]
            prob   = probs[mask]
            if y_true.size == 0:
                result[split_name] = {
                    'overall': {'acc': 0.0, 'auc': 50.0, 'f1': 0.0},
                    'sens_group': {},
                    'target_group': {},
                    'fairness': {'dp': 0.0, 'eo': 0.0},
                }
                continue
            pred   = (prob > 0.5).astype(int)

            acc_total = accuracy_score(y_true, pred) * 100
            auc_total = roc_auc_score(y_true, prob) * 100 if len(set(y_true)) == 2 else 50.0
            f1_total  = f1_score(y_true, pred, zero_division=0) * 100

            # Per sensitive group
            sens_metrics = {}
            for s in np.unique(sens):
                idx = sens == s
                sens_metrics[int(s)] = {
                    'acc': accuracy_score(y_true[idx], pred[idx]) * 100,
                    'auc': roc_auc_score(y_true[idx], prob[idx]) * 100
                          if len(np.unique(y_true[idx])) == 2 else 50.0,
                    'f1':  f1_score(y_true[idx], pred[idx], zero_division=0) * 100,
                }

            # Per target class
            y_metrics = {}
            for yval in np.unique(y_true):
                idx = y_true == yval
                y_metrics[int(yval)] = {
                    'acc': accuracy_score(y_true[idx], pred[idx]) * 100,
                    'auc': roc_auc_score((y_true == yval).astype(int),
                                         prob if yval == 1 else 1 - prob) * 100
                          if len(set(y_true)) == 2 else 50.0,
                    'f1':  f1_score((y_true == yval).astype(int),
                                    (pred == yval).astype(int), zero_division=0) * 100,
                }

            dp, eo = fair_metric(pred, y_true, sens)
            result[split_name] = {
                'overall':      {'acc': acc_total, 'auc': auc_total, 'f1': f1_total},
                'sens_group':   sens_metrics,
                'target_group': y_metrics,
                'fairness':     {'dp': dp * 100, 'eo': eo * 100},
            }

    for split_name in splits:
        accs[split_name]     = result[split_name]['overall']['acc']
        auc_rocs[split_name] = result[split_name]['overall']['auc']
        f1s[split_name]      = result[split_name]['overall']['f1']
        paritys[split_name]  = result[split_name]['fairness']['dp']
        equalitys[split_name]= result[split_name]['fairness']['eo']

    return accs, auc_rocs, paritys, equalitys
