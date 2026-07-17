from tqdm import tqdm
import numpy as np
import os
import sys
from types import SimpleNamespace
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
        build_initial_adaptation_state,
        class_conditional_mmd_loss,
        predict_target_proba,
        run_target_adaptation,
    )
except ImportError:
    from adaptation import (
        build_initial_adaptation_state,
        class_conditional_mmd_loss,
        predict_target_proba,
        run_target_adaptation,
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

    warmup_epochs  = max(1, int(args.train_epochs * 0.1))   # 10% of total epochs
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
        Group-balanced class prototype.  We first compute one normalized
        prototype for every (y,s) group, then average sensitive groups equally
        within each class, using all labeled nodes in V^so.

            p_y^S = Norm((1 / |S|) * sum_s p_{y,s}^S)

    p_ys  : {(y,s): [D]}
        Normalized Source class-group prototypes used by Eq. (8)/(15) to
        infer latent Target pseudo-sensitive memberships.

    Also saves model weights (backbone + cls_head) so the encoder can be
    reloaded without the original source data.

    Returns
    -------
    dict with keys: 'knowledge_version', 'p_y', 'p_ys', 'model_state'
    """
    # 不把源数据本身带到目标域，而是把源域中已经学到的统计结构导出来：
    #
    # - p_y:   类别原型，描述“某个类别大概在表示空间的什么位置”
    # - p_ys:  类别-敏感组原型，用于目标域伪敏感组推断
    #
    # 这样后续 target 适配阶段就不需要再访问 source 原始样本，
    # 满足 source-free domain adaptation 的设定。
    model.eval()
    with torch.no_grad():
        emb, _ = model(data.x, data.edge_index)          # [N, D]
        emb = emb.detach().cpu()

    y  = data.y.cpu()
    s  = data.sens_labels.cpu()
    # Eqs. (7)/(13) define the Source prototype bank over V^so, so all
    # labeled Source nodes contribute to the exported class-group statistics.
    emb_stats = emb
    y_stats   = y
    s_stats   = s

    # Eq. (7): construct normalized class-group prototypes first, then form
    # each class prototype by averaging groups equally.  This prevents the
    # numerically dominant sensitive group from determining the class center.
    group_prototypes = {}
    p_y = {}
    p_ys = {}
    for yv in [0, 1]:
        class_mask = y_stats == yv
        if class_mask.any():
            class_fallback = F.normalize(
                emb_stats[class_mask].mean(dim=0), p=2, dim=0, eps=1e-12
            ).cpu()
        else:
            class_fallback = torch.zeros(emb.shape[1], device='cpu')

        for sv in [0, 1]:
            mask = (y_stats == yv) & (s_stats == sv)
            if mask.any():
                group_prototype = F.normalize(
                    emb_stats[mask].mean(dim=0), p=2, dim=0, eps=1e-12
                ).cpu()
            else:
                group_prototype = class_fallback.clone()
            group_prototypes[(yv, sv)] = group_prototype
            p_ys[(yv, sv)] = group_prototype

        p_y[yv] = F.normalize(
            torch.stack(
                [group_prototypes[(yv, sv)] for sv in [0, 1]], dim=0
            ).mean(dim=0),
            p=2,
            dim=0,
            eps=1e-12,
        )

    return {
        'knowledge_version': 2,
        'p_y':        p_y,
        'p_ys':       p_ys,
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
        f"srcmeta={getattr(args, 'ablation', 'full') != 'metaalign'}",
        f"metalr={getattr(args, 'meta_lr', 0.01)}",
        f"lcoord={getattr(args, 'lambda_coord', 1.0)}",
        f"srcmmd={getattr(args, 'source_mmd_bandwidth', getattr(args, 'mmd_bandwidth', 1.0))}",
        f"srcmmdmin={getattr(args, 'source_mmd_min_samples', 2)}",
        f"srcmmdmax={getattr(args, 'source_mmd_max_samples', 0)}",
        "srcscope=all-vso",
        "knowledge=groupbank-v2",
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
    knowledge = ckpt.get('knowledge', {})
    if int(knowledge.get('knowledge_version', 0)) < 2 or 'p_ys' not in knowledge:
        print('[Checkpoint] obsolete Source knowledge format; retraining required.')
        return None, None
    # Reconstruct model
    model = FairGNN(args, encoder_type=args.inter_encoder).to(args.device)
    model.load_state_dict({k: v.to(args.device) for k, v in ckpt['model_state'].items()})
    return model, knowledge


def _build_source_free_eval_state(args, model, target_data, knowledge):
    """
    构造适配前的冷启动评估状态，和 SFDA 第一次前向的口径保持一致。
    """
    del model, target_data
    return build_initial_adaptation_state(args, knowledge)


# SFDA Target Adaptation
def adapt_target(args, target_data, knowledge):
    """Load the frozen source model and run FairMAC target adaptation."""
    device = args.device
    model = FairGNN(args, encoder_type=args.inter_encoder).to(device)
    model.load_state_dict({
        key: value.to(device)
        for key, value in knowledge['model_state'].items()
    })
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    target_view = SimpleNamespace(
        x=target_data.x.to(device),
        edge_index=target_data.edge_index.to(device),
    )
    state = run_target_adaptation(args, model, target_view, knowledge)
    return model, state


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

    source_class_labels = source_data.y.long().to(args.device)
    cls_labels  = source_class_labels.float()
    sens_labels = source_data.sens_labels.to(args.device)
    source_node_mask = torch.ones_like(source_class_labels, dtype=torch.bool)

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

            # Meta-coordination is source-only.  Following Eqs. (4)-(6), the
            # virtual step follows the fairness gradient and classification is
            # evaluated at the virtually updated Source-model parameters.
            meta_lr = float(getattr(args, 'meta_lr', 0.01))
            lambda_coord = float(getattr(args, 'lambda_coord', 1.0))
            use_source_metaalign = getattr(args, 'ablation', 'full') != 'metaalign'
            source_mmd_bandwidth = float(getattr(
                args,
                'source_mmd_bandwidth',
                getattr(args, 'mmd_bandwidth', 1.0),
            ))
            source_mmd_min_samples = max(
                1, int(getattr(args, 'source_mmd_min_samples', 2))
            )
            source_mmd_max_samples = max(
                0, int(getattr(args, 'source_mmd_max_samples', 0))
            )

            for epoch in tqdm(range(args.train_epochs), desc=f'Run {run_idx} [train]', leave=False):
                model.train()

                if use_source_metaalign:
                    source_cpu_rng_state = torch.random.get_rng_state()
                    source_cuda_rng_states = (
                        torch.cuda.get_rng_state_all()
                        if source_data.x.is_cuda
                        else None
                    )

                source_embedding, cls_logit = model(
                    source_data.x, source_data.edge_index
                )

                # Eqs. (3)-(6) define Source learning over all nodes in V^so.
                L_cls = criterion(cls_logit.view(-1), cls_labels)

                # Eq. (3): class-conditional fairness loss at theta, averaged
                # over the task classes present in the complete Source graph.
                L_mmd, valid_source_mmd_classes = class_conditional_mmd_loss(
                    source_embedding,
                    source_class_labels,
                    sens_labels.long(),
                    source_node_mask,
                    bandwidth=source_mmd_bandwidth,
                    chunk_size=getattr(args, 'mmd_chunk_size', 1024),
                    min_samples=source_mmd_min_samples,
                    max_samples=source_mmd_max_samples,
                    reduction='mean',
                )

                if use_source_metaalign:
                    # Preserve randomness consumed by both the ordinary
                    # forward and random MMD subsampling.  The virtual forward
                    # below must not rewind either part of the training stream.
                    post_source_step_cpu_rng_state = torch.random.get_rng_state()
                    post_source_step_cuda_rng_states = (
                        torch.cuda.get_rng_state_all()
                        if source_data.x.is_cuda
                        else None
                    )
                    named_model_parameters = list(model.named_parameters())
                    model_parameters = [
                        parameter for _, parameter in named_model_parameters
                    ]
                    grad_fair = torch.autograd.grad(
                        L_mmd,
                        model_parameters,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    # Eq. (4): virtually update the complete Source model.
                    # L_fair does not depend on the classifier, so its virtual
                    # gradient is zero and the head remains unchanged.
                    virtual_parameters = {
                        name: (
                            parameter
                            if gradient is None
                            else parameter - meta_lr * gradient
                        )
                        for (name, parameter), gradient in zip(
                            named_model_parameters, grad_fair
                        )
                    }
                    # Replay the exact RNG state used by the ordinary forward,
                    # so L_cls(theta') - L_cls(theta) is not contaminated by
                    # a different dropout mask.  Restore the RNG state after
                    # MMD sampling so the virtual pass consumes no additional
                    # randomness from the training stream.
                    torch.random.set_rng_state(source_cpu_rng_state)
                    if source_cuda_rng_states is not None:
                        torch.cuda.set_rng_state_all(source_cuda_rng_states)
                    try:
                        _, virtual_cls_logit = functional_call(
                            model,
                            virtual_parameters,
                            (source_data.x, source_data.edge_index),
                        )
                    finally:
                        torch.random.set_rng_state(post_source_step_cpu_rng_state)
                        if post_source_step_cuda_rng_states is not None:
                            torch.cuda.set_rng_state_all(
                                post_source_step_cuda_rng_states
                            )
                    virtual_cls_loss = criterion(
                        virtual_cls_logit.view(-1),
                        cls_labels,
                    )
                    # Eq. (5): coordination term L_cls(theta') - L_cls(theta).
                    L_coord = virtual_cls_loss - L_cls
                else:
                    L_coord = L_cls * 0.0

                # Eq. (6): L_meta = L_cls + beta L_fair + gamma L_coord.
                loss = (
                    L_cls
                    + args.lambda_fair * L_mmd
                    + lambda_coord * L_coord
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (epoch + 1) % 50 == 0:
                    print(f"[Run {run_idx} | Epoch {epoch+1:03d}] "
                          f"L_cls={L_cls.item():.4f}  "
                          f"L_mmd={L_mmd.item():.4f}  "
                          f"L_coord={L_coord.item():.4f}  "
                          f"MMD_cls={valid_source_mmd_classes}  "
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

        # Build the cold state without touching Target annotations.  Its
        # metrics are deliberately deferred until adaptation has finished.
        cold_state = _build_source_free_eval_state(args, model, target_data, knowledge)

        # ── Phase 2: SFDA adapt ────────────────────────────────────────────
        # knowledge was either loaded from checkpoint or extracted above
        configured_target_seed = getattr(args, 'target_seed', None)
        effective_target_seed = int(
            configured_target_seed
            if configured_target_seed is not None
            else getattr(args, 'seed', 1111) + run_idx
        )
        seed_everything(effective_target_seed)
        adapted_model, state = adapt_target(args, target_data, knowledge)

        # ── Final target evaluation ─────────────────────────────────────────
        # Target labels and sensitive attributes are first accessed here,
        # after all adaptation updates have completed.
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
    accs, auc_rocs, f1s, paritys, equalitys = {}, {}, {}, {}, {}

    encoder.eval()
    with torch.no_grad():
        feat, q_y = predict_target_proba(args, data, encoder, state)
        
        # extract positive probability
        probs = q_y[:, 1].cpu().numpy()

        # Target labels and sensitive attributes are accessed only after model
        # inference, exclusively for reporting predictive/fairness metrics.
        sens_labels = data.sens_labels
        test_labels = data.y[data.test_mask]
        t_idx_s0 = sens_labels[data.test_mask] == 0
        t_idx_s1 = sens_labels[data.test_mask] == 1
        t_idx_s0_y1 = torch.logical_and(t_idx_s0, test_labels == 1)
        t_idx_s1_y1 = torch.logical_and(t_idx_s1, test_labels == 1)
        t_idx_s0_y0 = torch.logical_and(t_idx_s0, test_labels == 0)
        t_idx_s1_y0 = torch.logical_and(t_idx_s1, test_labels == 0)

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
