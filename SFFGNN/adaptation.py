"""Source-free target adaptation with latent sensitive-group regularisation.

The target path deliberately consumes only ``x`` and ``edge_index``.  Source
class prototypes, source sensitive residuals, and the frozen source joint prior
are used to infer a latent joint posterior q(y, s) on target nodes.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_VERSION = 3
INFERENCE_STRATEGY = "latent_sensitive_marginalized"
JOINT_KEYS = ((0, 0), (0, 1), (1, 0), (1, 1))
ABLATION_MODES = ("full", "metaalign", "bca", "target_mmd", "residual")


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x), p=2, dim=dim, eps=1e-12)


def _graph_zero(x: torch.Tensor) -> torch.Tensor:
    """Return a differentiable zero connected to ``x``."""
    return x.sum() * 0.0


def safe_mmd_loss(
    h: torch.Tensor,
    s: torch.Tensor,
    bandwidth: float = 1.0,
    chunk_size: int = 1024,
    min_samples: int = 2,
    max_samples: int = 0,
) -> torch.Tensor:
    """Numerically safe biased RBF-MMD for two binary groups."""
    s = s.long().view(-1)
    h0 = h[s == 0]
    h1 = h[s == 1]
    if h0.shape[0] < min_samples or h1.shape[0] < min_samples:
        return _graph_zero(h)

    cap = max(int(max_samples), 0)
    if cap > 0:
        if h0.shape[0] > cap:
            indices = torch.linspace(0, h0.shape[0] - 1, cap, device=h0.device).long()
            h0 = h0[indices]
        if h1.shape[0] > cap:
            indices = torch.linspace(0, h1.shape[0] - 1, cap, device=h1.device).long()
            h1 = h1[indices]

    sigma = max(float(bandwidth), 1e-6)
    block = max(int(chunk_size), 1)

    def kernel_mean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = _graph_zero(x)
        count = 0
        denom = 2.0 * sigma * sigma
        for i in range(0, x.shape[0], block):
            xb = x[i : i + block]
            x2 = xb.square().sum(1, keepdim=True)
            for j in range(0, y.shape[0], block):
                yb = y[j : j + block]
                y2 = yb.square().sum(1).unsqueeze(0)
                dist2 = (x2 + y2 - 2.0 * xb @ yb.t()).clamp_min(0.0)
                total = total + torch.exp(-dist2 / denom).sum()
                count += xb.shape[0] * yb.shape[0]
        return total / max(count, 1)

    value = kernel_mean(h0, h0) + kernel_mean(h1, h1) - 2.0 * kernel_mean(h0, h1)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


class TargetAdapter(nn.Module):
    """Zero-initialised residual bottleneck applied after the frozen encoder."""

    def __init__(self, dim: int, bottleneck_dim: Optional[int] = None):
        super().__init__()
        hidden = max(1, int(bottleneck_dim or max(1, dim // 4)))
        self.dim = int(dim)
        self.bottleneck_dim = hidden
        self.down = nn.Linear(self.dim, hidden)
        self.up = nn.Linear(hidden, self.dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, base_embedding: torch.Tensor) -> torch.Tensor:
        delta = self.up(F.gelu(self.down(base_embedding)))
        return _safe_normalize(base_embedding + delta, dim=1)

    def config(self) -> Dict[str, int]:
        return {"dim": self.dim, "bottleneck_dim": self.bottleneck_dim}


def _source_class_prototypes(knowledge: Mapping, device: torch.device) -> torch.Tensor:
    values = []
    dim = int(knowledge["p_y"][0].numel())
    for class_id in (0, 1):
        value = knowledge["p_y"].get(class_id)
        if value is None or not torch.isfinite(value).all() or value.norm() < 1e-12:
            value = torch.zeros(dim)
            value[class_id % dim] = 1.0
        # Keep the exported source mean unchanged here because r_{y,s} was
        # defined as p_{y,s} - p_y; normalisation is applied after recombining
        # the class prototype and sensitive residual.
        values.append(torch.nan_to_num(value.to(device).float()))
    return torch.stack(values, dim=0)


def _source_sensitive_residuals(knowledge: Mapping, device: torch.device) -> torch.Tensor:
    dim = int(knowledge["p_y"][0].numel())
    residuals = torch.zeros((2, 2, dim), device=device)
    source_residuals = knowledge.get("r_ys", {})
    for class_id, group_id in JOINT_KEYS:
        value = source_residuals.get((class_id, group_id))
        if value is not None and torch.isfinite(value).all():
            residuals[class_id, group_id] = value.to(device).float()
    return residuals


def _source_joint_prior(knowledge: Mapping, device: torch.device) -> torch.Tensor:
    prior = torch.zeros((2, 2), device=device)
    source_prior = knowledge.get("pi_ys", {})
    for class_id, group_id in JOINT_KEYS:
        prior[class_id, group_id] = max(
            float(source_prior.get((class_id, group_id), 0.0)), 0.0
        )
    if prior.sum().item() <= 1e-12:
        prior.fill_(0.25)
    else:
        prior = prior / prior.sum().clamp_min(1e-8)
    return prior


def build_joint_prototypes(
    class_prototypes: torch.Tensor,
    source_residuals: torch.Tensor,
    use_sensitive_residual: bool = True,
) -> torch.Tensor:
    """Construct four target joint prototypes using fixed source residuals."""
    if use_sensitive_residual:
        joint = class_prototypes[:, None, :] + source_residuals
    else:
        joint = class_prototypes[:, None, :].expand(-1, 2, -1)
    return _safe_normalize(joint, dim=2)


def compute_joint_posterior(
    features: torch.Tensor,
    joint_prototypes: torch.Tensor,
    joint_prior: torch.Tensor,
    temperature: float,
    prior_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute q(y,s) over all four states and marginalise s to obtain q(y)."""
    features = _safe_normalize(features, dim=1)
    flattened_prototypes = _safe_normalize(joint_prototypes, dim=2).reshape(
        4, features.shape[1]
    )
    logits = features @ flattened_prototypes.t()
    logits = logits / max(float(temperature), 1e-6)
    flattened_prior = joint_prior.reshape(4).clamp_min(1e-8)
    logits = logits + float(prior_weight) * torch.log(flattened_prior).unsqueeze(0)
    q_ys = torch.softmax(torch.nan_to_num(logits), dim=1).reshape(-1, 2, 2)
    q_y = q_ys.sum(dim=2)
    return q_ys, q_y


def infer_latent_labels(
    q_ys: torch.Tensor,
    q_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer pseudo class and latent pseudo-sensitive labels from q(y,s)."""
    class_confidence, pseudo_y = q_y.max(dim=1)
    row_index = torch.arange(q_y.shape[0], device=q_y.device)
    conditional_group = q_ys[row_index, pseudo_y]
    conditional_group = conditional_group / conditional_group.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
    group_confidence, pseudo_s = conditional_group.max(dim=1)
    return class_confidence, pseudo_y, group_confidence, pseudo_s


def select_high_confidence(
    confidence: torch.Tensor,
    keep_fraction: float,
) -> torch.Tensor:
    if confidence.numel() == 0:
        return torch.zeros_like(confidence, dtype=torch.bool)
    fraction = min(max(float(keep_fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return torch.zeros_like(confidence, dtype=torch.bool)
    keep = max(1, int(round(fraction * confidence.numel())))
    indices = torch.topk(confidence, k=min(keep, confidence.numel())).indices
    mask = torch.zeros_like(confidence, dtype=torch.bool)
    mask[indices] = True
    return mask


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if values.shape[0] == 0:
        return fallback
    weights = torch.nan_to_num(weights, nan=0.0).clamp_min(0.0)
    denominator = weights.sum()
    if denominator.detach().item() <= 1e-8:
        return fallback
    return (values * weights.unsqueeze(1)).sum(0) / denominator.clamp_min(1e-8)


def build_class_prototypes(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    confidence: torch.Tensor,
    selected: torch.Tensor,
    previous: torch.Tensor,
) -> torch.Tensor:
    """Update class prototypes without using target sensitive attributes."""
    prototypes = []
    for class_id in (0, 1):
        mask = selected & (pseudo_labels == class_id)
        prototype = _weighted_mean(
            features[mask],
            confidence[mask],
            previous[class_id],
        )
        if not torch.isfinite(prototype).all() or prototype.detach().norm().item() < 1e-12:
            prototype = previous[class_id]
        prototypes.append(_safe_normalize(prototype, dim=0))
    return torch.stack(prototypes, dim=0)


def class_conditional_mmd_loss(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    group_labels: torch.Tensor,
    selected: torch.Tensor,
    bandwidth: float,
    chunk_size: int,
    min_samples: int,
    max_samples: int,
) -> Tuple[torch.Tensor, int]:
    """Shared class-conditional MMD for source and target representations.

    Source passes observed class/sensitive labels with ``train_mask``; target
    passes pseudo class/latent-sensitive labels with its confidence selection.
    """
    class_labels = class_labels.long().view(-1)
    group_labels = group_labels.long().view(-1)
    selected = selected.bool().view(-1)
    if not (
        features.shape[0]
        == class_labels.shape[0]
        == group_labels.shape[0]
        == selected.shape[0]
    ):
        raise ValueError(
            "features, class_labels, group_labels, and selected must have "
            "the same number of nodes"
        )

    loss = _graph_zero(features)
    valid_classes = 0
    selected_classes = torch.unique(class_labels[selected]).tolist()
    for class_id in selected_classes:
        class_mask = selected & (class_labels == int(class_id))
        group0_count = int((class_mask & (group_labels == 0)).sum().item())
        group1_count = int((class_mask & (group_labels == 1)).sum().item())
        if group0_count < min_samples or group1_count < min_samples:
            continue
        loss = loss + safe_mmd_loss(
            features[class_mask],
            group_labels[class_mask],
            bandwidth=bandwidth,
            chunk_size=chunk_size,
            min_samples=min_samples,
            max_samples=max_samples,
        )
        valid_classes += 1
    return loss, valid_classes


def _cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _fairness_weight(step: int, total_steps: int, warmup_ratio: float) -> float:
    warmup_steps = int(max(total_steps, 1) * max(float(warmup_ratio), 0.0))
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def run_target_adaptation(
    args,
    source_model: nn.Module,
    target_data,
    knowledge: Mapping,
) -> Dict:
    """Adapt a frozen source model using target x/edge_index and source knowledge."""
    ablation = str(getattr(args, "ablation", "full"))
    if ablation not in ABLATION_MODES:
        raise ValueError(
            "ablation must be one of {}; got {!r}".format(ABLATION_MODES, ablation)
        )

    use_bca = ablation != "bca"
    use_target_mmd = ablation != "target_mmd"
    use_sensitive_residual = ablation != "residual"

    device = args.device
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    with torch.no_grad():
        base_embedding, _ = source_model(target_data.x, target_data.edge_index)
        base_embedding = _safe_normalize(base_embedding.detach(), dim=1)

    dim = base_embedding.shape[1]
    configured_bottleneck = getattr(args, "adapter_bottleneck", None)
    bottleneck = int(
        configured_bottleneck
        if configured_bottleneck is not None
        else max(1, dim // 4)
    )
    adapter = TargetAdapter(dim, bottleneck).to(device)
    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=float(getattr(args, "adapt_lr", 1e-3)),
    )

    source_class_prototypes = _source_class_prototypes(knowledge, device)
    source_residuals = _source_sensitive_residuals(knowledge, device)
    source_joint_prior = _source_joint_prior(knowledge, device)
    class_prototypes = source_class_prototypes.detach().clone()

    epochs = max(1, int(getattr(args, "adapt_epochs", 200)))
    keep_fraction = float(getattr(args, "tau_c", 0.2))
    temperature = float(getattr(args, "proto_temp", 0.5))
    configured_prior_weight = float(getattr(args, "lambda_pi", 1.0)) * float(
        getattr(args, "tau_adjust", 1.0)
    )
    prior_weight = configured_prior_weight if use_bca else 0.0
    alpha_p = min(max(float(getattr(args, "alpha_p", 0.9)), 0.0), 1.0)
    lambda_cls = float(getattr(args, "lambda_s", 1.0))
    lambda_ent = float(getattr(args, "lambda_e", 0.1))
    lambda_anchor = float(getattr(args, "lambda_anchor", 1.0))
    lambda_fair = float(getattr(args, "target_lambda_fair", 1.0))
    bandwidth = float(
        getattr(
            args,
            "target_mmd_bandwidth",
            getattr(args, "mmd_bandwidth", 1.0),
        )
    )
    chunk_size = int(getattr(args, "mmd_chunk_size", 1024))
    min_mmd_samples = max(2, int(getattr(args, "target_mmd_min_samples", 2)))
    max_mmd_samples = max(0, int(getattr(args, "target_mmd_max_samples", 1024)))
    warmup_ratio = float(getattr(args, "fairness_warmup_ratio", 0.1))

    mmd_steps = 0
    mmd_valid_class_steps = 0
    last_losses = {}
    max_adapter_grad_norm = 0.0
    selected_count = 0
    last_group_confidence = 0.0

    print(
        "[FairMAC] ablation={} epochs={} adapter={} bottleneck={} "
        "prior=frozen target_mmd={}".format(
            ablation, epochs, dim, bottleneck, use_target_mmd
        )
    )

    for step in range(epochs):
        adapter.train()
        student_features = adapter(base_embedding)

        with torch.no_grad():
            reference_joint_prototypes = build_joint_prototypes(
                class_prototypes,
                source_residuals,
                use_sensitive_residual=use_sensitive_residual,
            )
            reference_q_ys, reference_q_y = compute_joint_posterior(
                student_features.detach(),
                reference_joint_prototypes,
                source_joint_prior,
                temperature,
                prior_weight,
            )
            confidence, pseudo_labels, group_confidence, pseudo_sensitive = (
                infer_latent_labels(reference_q_ys, reference_q_y)
            )
            selected = select_high_confidence(confidence, keep_fraction)

        selected_count = int(selected.sum().item())
        if selected.any():
            last_group_confidence = float(group_confidence[selected].mean().item())

        current_class_prototypes = build_class_prototypes(
            student_features,
            pseudo_labels,
            confidence,
            selected,
            class_prototypes,
        )
        current_joint_prototypes = build_joint_prototypes(
            current_class_prototypes,
            source_residuals,
            use_sensitive_residual=use_sensitive_residual,
        )
        _, current_q_y = compute_joint_posterior(
            student_features,
            current_joint_prototypes,
            source_joint_prior,
            temperature,
            prior_weight,
        )
        current_q_y = current_q_y.clamp_min(1e-8)

        if selected.any():
            classification_loss = F.nll_loss(
                current_q_y[selected].log(),
                pseudo_labels[selected],
            )
        else:
            classification_loss = _graph_zero(current_q_y)
        entropy_loss = -(current_q_y * current_q_y.log()).sum(1).mean()

        if use_target_mmd:
            mmd_loss, valid_mmd_classes = class_conditional_mmd_loss(
                student_features,
                pseudo_labels,
                pseudo_sensitive,
                selected,
                bandwidth=bandwidth,
                chunk_size=chunk_size,
                min_samples=min_mmd_samples,
                max_samples=max_mmd_samples,
            )
            if valid_mmd_classes > 0:
                mmd_steps += 1
                mmd_valid_class_steps += valid_mmd_classes
        else:
            mmd_loss = _graph_zero(student_features)
            valid_mmd_classes = 0

        ramp = _fairness_weight(step, epochs, warmup_ratio)
        fairness_loss = ramp * lambda_fair * mmd_loss
        feature_anchor = F.mse_loss(student_features, base_embedding)
        prototype_anchor = F.mse_loss(
            current_class_prototypes,
            source_class_prototypes,
        )
        anchor_loss = feature_anchor + 0.25 * prototype_anchor

        total_loss = (
            lambda_cls * classification_loss
            + lambda_ent * entropy_loss
            + lambda_anchor * anchor_loss
            + fairness_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm_sq = 0.0
        for parameter in adapter.parameters():
            if parameter.grad is not None:
                grad_norm_sq += float(parameter.grad.detach().square().sum().item())
        grad_norm = grad_norm_sq ** 0.5
        max_adapter_grad_norm = max(max_adapter_grad_norm, grad_norm)
        optimizer.step()

        with torch.no_grad():
            class_prototypes = _safe_normalize(
                alpha_p * class_prototypes
                + (1.0 - alpha_p) * current_class_prototypes.detach(),
                dim=1,
            )

        last_losses = {
            "total": float(total_loss.detach().item()),
            "classification": float(classification_loss.detach().item()),
            "entropy": float(entropy_loss.detach().item()),
            "anchor": float(anchor_loss.detach().item()),
            "mmd": float(mmd_loss.detach().item()),
        }
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == epochs:
            print(
                "  [Adapt {:3d}] hc={}/{} cls={:.4f} mmd={:.4f} "
                "mmd_cls={} anchor={:.4f} grad={:.4e}".format(
                    step,
                    selected_count,
                    base_embedding.shape[0],
                    last_losses["classification"],
                    last_losses["mmd"],
                    valid_mmd_classes,
                    last_losses["anchor"],
                    grad_norm,
                )
            )

    adapter.eval()
    with torch.no_grad():
        final_features = adapter(base_embedding)
        provisional_joint_prototypes = build_joint_prototypes(
            class_prototypes,
            source_residuals,
            use_sensitive_residual=use_sensitive_residual,
        )
        final_q_ys, final_q_y = compute_joint_posterior(
            final_features,
            provisional_joint_prototypes,
            source_joint_prior,
            temperature,
            prior_weight,
        )
        final_confidence, final_pseudo, final_group_confidence, _ = infer_latent_labels(
            final_q_ys, final_q_y
        )
        final_selected = select_high_confidence(final_confidence, keep_fraction)
        candidate_class_prototypes = build_class_prototypes(
            final_features,
            final_pseudo,
            final_confidence,
            final_selected,
            class_prototypes,
        )
        final_class_prototypes = _safe_normalize(
            alpha_p * class_prototypes
            + (1.0 - alpha_p) * candidate_class_prototypes,
            dim=1,
        )
        final_joint_prototypes = build_joint_prototypes(
            final_class_prototypes,
            source_residuals,
            use_sensitive_residual=use_sensitive_residual,
        )

    source_pi_ys = {
        (class_id, group_id): float(source_joint_prior[class_id, group_id].item())
        for class_id, group_id in JOINT_KEYS
    }
    source_r_ys = {
        (class_id, group_id): source_residuals[class_id, group_id]
        .detach()
        .cpu()
        .clone()
        for class_id, group_id in JOINT_KEYS
    }
    state = {
        "state_version": STATE_VERSION,
        "ablation": ablation,
        "source_metaalign_enabled": ablation != "metaalign",
        "use_bca": use_bca,
        "use_target_mmd": use_target_mmd,
        "use_sensitive_residual": use_sensitive_residual,
        "source_prior_frozen": True,
        "adapter_config": adapter.config(),
        "adapter_state_dict": _cpu_state_dict(adapter),
        "class_prototypes": final_class_prototypes.detach().cpu().clone(),
        "joint_prototypes": final_joint_prototypes.detach().cpu().clone(),
        "joint_prior": source_joint_prior.detach().cpu().clone(),
        "inference_strategy": INFERENCE_STRATEGY,
        "proto_temp": temperature,
        "lambda_pi": float(getattr(args, "lambda_pi", 1.0)) if use_bca else 0.0,
        "tau_adjust": float(getattr(args, "tau_adjust", 1.0)),
        "diagnostics": {
            "mmd_steps": mmd_steps,
            "mmd_valid_class_steps": mmd_valid_class_steps,
            "last_losses": last_losses,
            "adapter_grad_norm_max": max_adapter_grad_norm,
            "source_encoder_frozen": all(
                not parameter.requires_grad for parameter in source_model.parameters()
            ),
            "selected_count": int(final_selected.sum().item()),
            "latent_group_confidence": float(
                final_group_confidence[final_selected].mean().item()
                if final_selected.any()
                else last_group_confidence
            ),
            "prior_shift": 0.0,
        },
        # Compatibility metadata for analysis code; all values remain source-derived.
        "p_y": {
            class_id: final_class_prototypes[class_id].detach().cpu().clone()
            for class_id in (0, 1)
        },
        "r_ys": source_r_ys,
        "pi_ys": source_pi_ys,
        "final_protos": {
            (class_id, group_id): final_joint_prototypes[class_id, group_id]
            .detach()
            .cpu()
            .clone()
            for class_id, group_id in JOINT_KEYS
        },
    }
    return state


def build_adapter_from_state(state: Mapping, device: torch.device) -> TargetAdapter:
    config = state["adapter_config"]
    adapter = TargetAdapter(
        int(config["dim"]),
        int(config["bottleneck_dim"]),
    ).to(device)
    adapter.load_state_dict(
        {key: value.to(device) for key, value in state["adapter_state_dict"].items()}
    )
    adapter.eval()
    return adapter


def predict_from_adaptation_state(
    base_embedding: torch.Tensor,
    state: Mapping,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(state.get("state_version", 0)) < STATE_VERSION:
        raise ValueError(
            "Legacy adaptation states used target sensitive attributes; regenerate "
            "the state with the source-free version-3 pipeline"
        )
    if state.get("inference_strategy") != INFERENCE_STRATEGY:
        raise ValueError("The adaptation state uses an unsupported inference strategy")

    device = base_embedding.device
    adapter = build_adapter_from_state(state, device)
    features = adapter(_safe_normalize(base_embedding, dim=1))
    joint_prototypes = state["joint_prototypes"].to(device)
    joint_prior = state["joint_prior"].to(device)
    prior_weight = float(state.get("lambda_pi", 1.0)) * float(
        state.get("tau_adjust", 1.0)
    )
    _, probabilities = compute_joint_posterior(
        features,
        joint_prototypes,
        joint_prior,
        float(state.get("proto_temp", 1.0)),
        prior_weight,
    )
    return features, probabilities


def predict_target_proba(
    args,
    data,
    encoder,
    state: Mapping,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return adapted features and class probabilities without target y/s."""
    del args
    encoder.eval()
    with torch.no_grad():
        base_embedding, _ = encoder(data.x, data.edge_index)
        return predict_from_adaptation_state(base_embedding, state)


def save_adaptation_state(state: Mapping, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(dict(state), path)


def load_adaptation_state(path: str, map_location="cpu") -> Dict:
    return torch.load(path, map_location=map_location)
