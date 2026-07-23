import hashlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.func import functional_call
from tqdm import tqdm

from learn import evaluate_per_class
from models import FairGNN
from utils import fair_metric, seed_everything

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


def _init_fair_gnn(args):
    """Build the source model, optimizer, and warm-up/cosine scheduler."""
    model = FairGNN(args, encoder_type=args.inter_encoder).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.lr2_reg,
    )

    warmup_epochs = max(1, int(args.train_epochs * 0.1))
    cosine_epochs = args.train_epochs - warmup_epochs

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=args.lr / 100,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return model, optimizer, scheduler


def _disentanglement_loss(
    task_view,
    sensitive_view,
    temperature,
    batch_size,
):
    """Paper Eq. (5), evaluated on one uniformly sampled source mini-batch.

    The paper minimizes the mean log-probability of each task view matching
    its paired sensitive view.  This is the negative of ordinary cross-entropy:
    minimizing it suppresses paired-view identifiability instead of promoting
    it.
    """
    node_count = task_view.shape[0]
    if node_count == 0:
        return task_view.sum() * 0.0
    temperature = float(temperature)
    batch_size = int(batch_size)
    if temperature <= 0.0:
        raise ValueError("disentanglement temperature must be positive")
    if batch_size < 1:
        raise ValueError("disentanglement batch size must be positive")
    size = min(batch_size, node_count)
    if size < node_count:
        indices = torch.randperm(node_count, device=task_view.device)[:size]
        task_view = task_view[indices]
        sensitive_view = sensitive_view[indices]
    normalized_task = F.normalize(task_view, p=2, dim=1, eps=1e-12)
    normalized_sensitive = F.normalize(
        sensitive_view,
        p=2,
        dim=1,
        eps=1e-12,
    )
    pairwise_logits = normalized_task @ normalized_sensitive.t()
    pairwise_logits = pairwise_logits / temperature
    paired_log_probabilities = F.log_softmax(pairwise_logits, dim=1).diagonal()
    return paired_log_probabilities.mean()


def _normalized_masked_mean(features, mask, name):
    if not bool(mask.any()):
        raise ValueError(f"Cannot construct {name}: the source subset is empty")
    mean = features[mask].mean(dim=0)
    if not bool(torch.isfinite(mean).all()) or mean.norm().item() <= 1e-12:
        raise ValueError(f"Cannot construct {name}: the source mean is invalid")
    return F.normalize(mean, p=2, dim=0, eps=1e-12)


def extract_source_knowledge(data, model):
    """
    Export the source-free package defined by Eqs. (10)-(11).

    p_y   : {0: [D], 1: [D]}
        Group-balanced class prototype.  We first compute one normalized
        prototype for every (y,s) group, then average sensitive groups equally
        within each class, using all labeled nodes in V^so.

            p_y^S = Norm((1 / |S|) * sum_s p_{y,s}^S)

    a_g   : {0: [D], 1: [D]}
        Class-agnostic group anchors in the decoupled sensitive space.

    The exported model retains both projection heads and the task classifier,
    while the source-only sensitive classifier is discarded.

    The result contains ``knowledge_version``, ``p_y``, ``a_g``, and the
    exported source-model state.
    """
    model.eval()
    with torch.no_grad():
        task_view, sensitive_view = model.encode_views(
            data.x,
            data.edge_index,
        )
        task_view = task_view.detach().cpu()
        sensitive_view = sensitive_view.detach().cpu()

    y = data.y.cpu().long()
    s = data.sens_labels.cpu().long()

    p_y = {}
    for yv in (0, 1):
        group_prototypes = []
        for sv in (0, 1):
            mask = (y == yv) & (s == sv)
            group_prototypes.append(
                _normalized_masked_mean(
                    task_view,
                    mask,
                    f"task prototype for class={yv}, group={sv}",
                )
            )
        p_y[yv] = _normalized_masked_mean(
            torch.stack(group_prototypes),
            torch.ones(2, dtype=torch.bool),
            f"group-balanced task prototype for class={yv}",
        )

    a_g = {}
    for sv in (0, 1):
        a_g[sv] = _normalized_masked_mean(
            sensitive_view,
            s == sv,
            f"sensitive anchor for group={sv}",
        )

    exported_state = {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("sens_head.")
    }
    return {
        "knowledge_version": 3,
        "p_y": {key: value.cpu() for key, value in p_y.items()},
        "a_g": {key: value.cpu() for key, value in a_g.items()},
        "model_state": exported_state,
    }


def _get_checkpoint_name(args, run_idx):
    """Return a short, deterministic name for the complete source config."""
    name_parts = [
        f"ds={args.dataset}",
        f"src={args.inid}",
        f"enc={args.inter_encoder}",
        f"hid={args.hidden_dim}",
        f"lay={args.n_layers}",
        f"ep={args.train_epochs}",
        f"lr={args.lr}",
        f"wd={args.lr2_reg}",
        f"drop={args.dropout}",
        f"lfair={args.lambda_fair}",
        f"srcmeta={getattr(args, 'ablation', 'full') != 'metaalign'}",
        f"metalr={getattr(args, 'meta_lr', 0.01)}",
        f"lcoord={getattr(args, 'lambda_coord', 1.0)}",
        f"lsen={getattr(args, 'lambda_sen', 1.0)}",
        f"ldec={getattr(args, 'lambda_dec', 1.0)}",
        f"taud={getattr(args, 'disentangle_temp', 0.1)}",
        f"decbatch={getattr(args, 'disentangle_batch_size', 512)}",
        f"srcmmd={getattr(args, 'source_mmd_bandwidth', getattr(args, 'mmd_bandwidth', 1.0))}",
        f"srcmmdmin={getattr(args, 'source_mmd_min_samples', 2)}",
        f"srcmmdmax={getattr(args, 'source_mmd_max_samples', 0)}",
        f"srcmmdchunk={getattr(args, 'mmd_chunk_size', 1024)}",
        "srcscope=all-labeled-vso",
        "architecture=dual-head-v1",
        "knowledge=ember-bank-v3",
        f"run={run_idx}",
        f"seed={args.seed}",
    ]
    signature = "|".join(name_parts)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    readable_prefix = "ember_{}_{}_{}_run{}_seed{}".format(
        args.dataset,
        str(args.inid).replace("/", "-").replace("\\", "-"),
        args.inter_encoder.lower(),
        run_idx,
        args.seed,
    )
    return f"{readable_prefix}_{digest}"


def _checkpoint_path(args, run_idx):
    """Return the full path for a checkpoint."""
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
    path = _checkpoint_path(args, run_idx)
    if not os.path.exists(path):
        return None, None
    print(f'[Checkpoint] loading ← {path}')
    ckpt = torch.load(path, map_location=args.device)
    knowledge = ckpt.get('knowledge', {})
    if int(knowledge.get('knowledge_version', 0)) < 3 or 'a_g' not in knowledge:
        print('[Checkpoint] obsolete Source knowledge format; retraining required.')
        return None, None
    model = FairGNN(args, encoder_type=args.inter_encoder).to(args.device)
    model.load_state_dict({k: v.to(args.device) for k, v in ckpt['model_state'].items()})
    return model, knowledge


def adapt_target(args, target_data, knowledge):
    """Load the frozen source package and run EMBER target adaptation."""
    device = args.device
    model = FairGNN(
        args,
        encoder_type=args.inter_encoder,
        include_sensitive_classifier=False,
    ).to(device)
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


def train_and_adapt(args, source_data, target_data):
    """
    For each run:
      1. Train FairGNN on source domain.
      2. Extract source knowledge.
      3. SFDA adapt to target domain.
      4. Evaluate source / target-before / target-after.
    """
    src_acc      = np.zeros([args.runs, 1])
    src_auc_roc  = np.zeros([args.runs, 1])
    src_parity   = np.zeros([args.runs, 1])
    src_equality = np.zeros([args.runs, 1])
    tgt_acc      = np.zeros([args.runs, 1])
    tgt_auc_roc  = np.zeros([args.runs, 1])
    tgt_parity   = np.zeros([args.runs, 1])
    tgt_equality = np.zeros([args.runs, 1])
    ada_acc      = np.zeros([args.runs, 1])
    ada_auc_roc  = np.zeros([args.runs, 1])
    ada_parity   = np.zeros([args.runs, 1])
    ada_equality = np.zeros([args.runs, 1])

    source_data = source_data.to(args.device)
    target_data = target_data.to(args.device)

    source_class_labels = source_data.y.long().to(args.device)
    cls_labels  = source_class_labels.float()
    sens_labels = source_data.sens_labels.long().to(args.device)
    sens_targets = sens_labels.float()
    # Source learning uses the complete graph, but label-dependent objectives
    # must only consume nodes with valid binary annotations.  Pokec keeps -1
    # as the missing-label sentinel so those nodes can still participate in
    # message passing without becoming invalid BCE targets or a third MMD
    # class.
    source_labeled_mask = (
        (source_class_labels == 0) | (source_class_labels == 1)
    )
    if not bool(source_labeled_mask.any()):
        raise ValueError("Source graph contains no valid binary labels")
    source_sensitive_mask = (sens_labels == 0) | (sens_labels == 1)
    if not bool(source_sensitive_mask.any()):
        raise ValueError("Source graph contains no valid binary sensitive labels")
    source_fair_mask = source_labeled_mask & source_sensitive_mask

    criterion = nn.BCEWithLogitsLoss()

    for run_idx in tqdm(range(args.runs), unit='run'):

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
            model, optimizer, scheduler = _init_fair_gnn(args)

            # Meta-coordination is source-only.  Following Eqs. (7)-(9), the
            # virtual step follows the fairness gradient and classification is
            # evaluated at the virtually updated Source-model parameters.
            meta_lr = float(getattr(args, 'meta_lr', 0.01))
            if meta_lr <= 0.0:
                raise ValueError("meta_lr (paper alpha) must be positive")
            configured_coordination_strength = float(
                getattr(args, 'lambda_coord', 1.0)
            )
            if not 0.0 <= configured_coordination_strength <= 1.0:
                raise ValueError(
                    "lambda_coord (paper gamma) must lie in [0, 1]; "
                    "gamma=1 is the full fairness update"
                )
            coordination_strength = (
                configured_coordination_strength
                if getattr(args, 'ablation', 'full') != 'metaalign'
                else 0.0
            )
            lambda_sen = float(getattr(args, 'lambda_sen', 1.0))
            lambda_dec = float(getattr(args, 'lambda_dec', 1.0))
            if lambda_sen < 0.0 or lambda_dec < 0.0:
                raise ValueError("lambda_sen and lambda_dec must be non-negative")
            disentangle_temp = float(getattr(args, 'disentangle_temp', 0.1))
            disentangle_batch_size = int(
                getattr(args, 'disentangle_batch_size', 512)
            )
            source_mmd_bandwidth = float(getattr(
                args,
                'source_mmd_bandwidth',
                getattr(args, 'mmd_bandwidth', 1.0),
            ))
            source_mmd_min_samples = int(
                getattr(args, 'source_mmd_min_samples', 2)
            )
            source_mmd_max_samples = int(
                getattr(args, 'source_mmd_max_samples', 0)
            )
            if source_mmd_min_samples < 1:
                raise ValueError("source_mmd_min_samples must be positive")
            if source_mmd_max_samples < 0:
                raise ValueError("source_mmd_max_samples must be non-negative")

            for epoch in tqdm(range(args.train_epochs), desc=f'Run {run_idx} [train]', leave=False):
                model.train()

                if coordination_strength > 0.0:
                    source_cpu_rng_state = torch.random.get_rng_state()
                    source_cuda_rng_states = (
                        torch.cuda.get_rng_state_all()
                        if source_data.x.is_cuda
                        else None
                    )

                (source_task_view, source_sensitive_view,
                 cls_logit, sens_logit) = model.source_forward(
                    source_data.x, source_data.edge_index
                )

                # Eqs. (3)-(6) consume the relevant valid annotations, while
                # every source node remains in the full-graph forward pass.
                L_cls = criterion(
                    cls_logit.view(-1)[source_labeled_mask],
                    cls_labels[source_labeled_mask],
                )
                L_sen = criterion(
                    sens_logit.view(-1)[source_sensitive_mask],
                    sens_targets[source_sensitive_mask],
                )
                L_dec = _disentanglement_loss(
                    source_task_view,
                    source_sensitive_view,
                    disentangle_temp,
                    disentangle_batch_size,
                )

                # Eq. (6): class-conditional inter-group MMD in task space.
                L_mmd, valid_source_mmd_classes = class_conditional_mmd_loss(
                    source_task_view,
                    source_class_labels,
                    sens_labels,
                    source_fair_mask,
                    bandwidth=source_mmd_bandwidth,
                    chunk_size=getattr(args, 'mmd_chunk_size', 1024),
                    min_samples=source_mmd_min_samples,
                    max_samples=source_mmd_max_samples,
                    reduction='mean',
                )

                if coordination_strength > 0.0:
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
                        # Paper Eq. (9) uses the first-order MAML
                        # approximation: the inner gradient is detached.
                        create_graph=False,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    # Eq. (9): theta' = theta - gamma * alpha * grad L_fair.
                    # L_fair does not depend on the classifier, so its virtual
                    # gradient is zero and the head remains unchanged.
                    virtual_parameters = {
                        name: (
                            parameter
                            if gradient is None
                            else parameter
                            - coordination_strength * meta_lr * gradient.detach()
                        )
                        for (name, parameter), gradient in zip(
                            named_model_parameters, grad_fair
                        )
                    }
                    # Replay the exact RNG state used by the ordinary forward,
                    # so the meta-test loss is not contaminated by a different
                    # dropout mask. Restore the post-loss RNG state afterward.
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
                        virtual_cls_logit.view(-1)[source_labeled_mask],
                        cls_labels[source_labeled_mask],
                    )
                else:
                    # Variant 1 / gamma=0 reduces Eq. (9) to direct
                    # classification plus class-conditional fair alignment.
                    virtual_cls_loss = L_cls

                # Eqs. (9) and (19): the classification term is evaluated at
                # theta'; sensitive identification and disentanglement are
                # source-only auxiliary branches.
                loss = (
                    virtual_cls_loss
                    + args.lambda_fair * L_mmd
                    + lambda_sen * L_sen
                    + lambda_dec * L_dec
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (epoch + 1) % 50 == 0:
                    print(f"[Run {run_idx} | Epoch {epoch+1:03d}] "
                          f"L_cls={L_cls.item():.4f}  "
                          f"L_mmd={L_mmd.item():.4f}  "
                          f"L_meta_cls={virtual_cls_loss.item():.4f}  "
                          f"L_sen={L_sen.item():.4f}  "
                          f"L_dec={L_dec.item():.4f}  "
                          f"MMD_cls={valid_source_mmd_classes}  "
                          f"Total={loss.item():.4f}")

            knowledge = extract_source_knowledge(source_data, model)
            _save_checkpoint(args, run_idx, model, knowledge)

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

        cold_state = build_initial_adaptation_state(args, knowledge)

        configured_target_seed = getattr(args, 'target_seed', None)
        effective_target_seed = int(
            configured_target_seed
            if configured_target_seed is not None
            else getattr(args, 'seed', 1111) + run_idx * 1111
        )
        seed_everything(effective_target_seed)
        adapted_model, state = adapt_target(args, target_data, knowledge)

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
    """Evaluate target probabilities with the paper's ACC, AUC, DP, and EO."""
    accs, auc_rocs, paritys, equalitys = {}, {}, {}, {}

    encoder.eval()
    with torch.no_grad():
        feat, q_y = predict_target_proba(args, data, encoder, state)
        probs = q_y[:, 1].cpu().numpy()

        y_all = data.y.cpu().numpy()
        sens_all = data.sens_labels.cpu().numpy()
        valid_label_mask = (data.y == 0) | (data.y == 1)
        all_mask = (data.train_mask | data.val_mask | data.test_mask) & valid_label_mask

        splits = {
            'all':   all_mask.cpu().numpy(),
            'train': (data.train_mask & valid_label_mask).cpu().numpy(),
            'val':   (data.val_mask & valid_label_mask).cpu().numpy(),
            'test':  (data.test_mask & valid_label_mask).cpu().numpy(),
        }

        export_y = data.y[all_mask]
        export_sens = data.sens_labels[all_mask]
        labels = torch.full_like(export_y, -1, dtype=torch.int64)
        labels[(export_y == 1) & (export_sens == 0)] = 0
        labels[(export_y == 1) & (export_sens == 1)] = 1
        labels[(export_y == 0) & (export_sens == 0)] = 2
        labels[(export_y == 0) & (export_sens == 1)] = 3
        export_embeddings = not getattr(args, 'disable_embedding_export', False)
        if export_embeddings:
            np.savez(f"{args.dataset}_feat.npz",
                     representations=feat[all_mask].cpu().numpy())
            np.savez(f"{args.dataset}_labels.npz",
                     labels=labels.cpu().numpy())
        if (export_embeddings and save_visualization
                and getattr(args, 'save_visualization_embeddings', False)):
            embeddings_root = os.path.join(PROJECT_ROOT, 'visualization', 'embeddings')
            save_visualization_embeddings(
                embeddings_root,
                'EMBER',
                args.dataset,
                feat[all_mask].cpu().numpy(),
                labels=labels.cpu().numpy(),
            )

        for split_name, mask in splits.items():
            y_true = y_all[mask]
            if y_true.size == 0:
                accs[split_name] = 0.0
                auc_rocs[split_name] = 50.0
                paritys[split_name] = 0.0
                equalitys[split_name] = 0.0
                continue

            sens = sens_all[mask]
            prob = probs[mask]
            pred = (prob > 0.5).astype(int)
            accs[split_name] = accuracy_score(y_true, pred) * 100
            auc_rocs[split_name] = (
                roc_auc_score(y_true, prob) * 100
                if len(np.unique(y_true)) == 2
                else 50.0
            )
            valid_sensitive = (sens == 0) | (sens == 1)
            dp, eo = fair_metric(
                pred[valid_sensitive],
                y_true[valid_sensitive],
                sens[valid_sensitive],
            )
            paritys[split_name] = dp * 100
            equalitys[split_name] = eo * 100

    return accs, auc_rocs, paritys, equalitys
