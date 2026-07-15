"""FairMAC target adaptation for source-free fair node classification.

This module deliberately depends only on PyTorch.  The source FairGNN remains
unchanged so existing source checkpoints and exported knowledge stay loadable.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


STATE_VERSION = 2
ABLATION_MODES = ("full", "metaalign", "bca", "prior_ema", "residual")
PRIOR_UPDATE_MODES = ("ema", "frozen", "replace", "cumulative")


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x), p=2, dim=dim, eps=1e-12)


def _graph_zero(x: torch.Tensor) -> torch.Tensor:
    """A differentiable zero connected to ``x``."""
    return x.sum() * 0.0


def safe_mmd_loss(
    h: torch.Tensor,
    s: torch.Tensor,
    bandwidth: float = 1.0,
    chunk_size: int = 1024,
    min_samples: int = 2,
    max_samples: int = 0,
) -> torch.Tensor:
    """Numerically safe biased RBF-MMD for two binary sensitive groups."""
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


def _initial_prototypes(knowledge: Mapping, device: torch.device) -> torch.Tensor:
    values = []
    dim = int(knowledge["p_y"][0].numel())
    for class_id in (0, 1):
        value = knowledge["p_y"].get(class_id)
        if value is None or not torch.isfinite(value).all() or value.norm() < 1e-12:
            value = torch.zeros(dim)
            value[class_id % dim] = 1.0
        values.append(_safe_normalize(value.to(device).float(), dim=0))
    return torch.stack(values, dim=0)


def _initial_soft_counts(knowledge: Mapping, device: torch.device, smoothing: float) -> torch.Tensor:
    counts = torch.full((2, 2), max(float(smoothing), 1e-6), device=device)
    pi_ys = knowledge.get("pi_ys", {})
    for class_id in (0, 1):
        for group_id in (0, 1):
            value = float(pi_ys.get((class_id, group_id), 0.0))
            if value > 0.0:
                counts[group_id, class_id] += value
    return counts


def priors_from_counts(counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-8)
    group_prior = counts / counts.sum(1, keepdim=True).clamp_min(1e-8)
    global_counts = counts.sum(0)
    global_prior = global_counts / global_counts.sum().clamp_min(1e-8)
    return group_prior, global_prior


def estimate_soft_counts(
    probabilities: torch.Tensor,
    sensitive: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    counts = probabilities.new_full((2, 2), max(float(smoothing), 1e-6))
    sensitive = sensitive.long().view(-1)
    detached = torch.nan_to_num(probabilities.detach(), nan=0.5).clamp_min(0.0)
    detached = detached / detached.sum(1, keepdim=True).clamp_min(1e-8)
    for group_id in (0, 1):
        mask = sensitive == group_id
        if mask.any():
            counts[group_id] += detached[mask].sum(0)
    return counts


def update_prior_counts(
    previous: torch.Tensor,
    current: torch.Tensor,
    mode: str,
    momentum: float,
    smoothing: float,
) -> torch.Tensor:
    """Update group/class soft counts using the selected online estimator."""
    if mode == "ema":
        return momentum * previous + (1.0 - momentum) * current
    if mode == "frozen":
        return previous
    if mode == "replace":
        return current
    if mode == "cumulative":
        raw_current = (current - max(float(smoothing), 1e-6)).clamp_min(0.0)
        return previous + raw_current
    raise ValueError(
        "prior_update_mode must be one of {}; got {!r}".format(
            PRIOR_UPDATE_MODES, mode
        )
    )


def compute_posterior(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    group_prior: torch.Tensor,
    sensitive: torch.Tensor,
    temperature: float,
    prior_weight: float,
) -> torch.Tensor:
    features = _safe_normalize(features, dim=1)
    prototypes = _safe_normalize(prototypes, dim=1)
    logits = features @ prototypes.t()
    logits = logits / max(float(temperature), 1e-6)
    sensitive = sensitive.long().clamp(0, 1)
    prior = group_prior[sensitive]
    logits = logits + float(prior_weight) * torch.log(prior.clamp_min(1e-8))
    return torch.softmax(torch.nan_to_num(logits), dim=1)


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


def build_prototypes(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    confidence: torch.Tensor,
    selected: torch.Tensor,
    sensitive: torch.Tensor,
    previous: torch.Tensor,
    use_sensitive_residual: bool = True,
) -> torch.Tensor:
    prototypes = []
    sensitive = sensitive.long()
    group_sizes = torch.stack([(sensitive == group_id).sum() for group_id in (0, 1)]).float()
    inverse_group = 1.0 / group_sizes.clamp_min(1.0)

    for class_id in (0, 1):
        class_mask = selected & (pseudo_labels == class_id)
        fallback = previous[class_id]
        if not class_mask.any():
            prototypes.append(fallback)
            continue

        class_features = features[class_mask]
        class_confidence = confidence[class_mask]
        confidence_proto = _weighted_mean(
            class_features,
            class_confidence,
            fallback,
        )

        if use_sensitive_residual:
            weights = inverse_group[sensitive[class_mask]]
            residual = _weighted_mean(
                class_features - confidence_proto.unsqueeze(0),
                weights,
                torch.zeros_like(confidence_proto),
            )
            prototype = confidence_proto + residual
        else:
            prototype = confidence_proto

        if not torch.isfinite(prototype).all() or prototype.detach().norm().item() < 1e-12:
            prototype = fallback
        prototypes.append(_safe_normalize(prototype, dim=0))

    return torch.stack(prototypes, dim=0)


def compute_target_fairness_loss(
    features: torch.Tensor,
    sensitive: torch.Tensor,
    bandwidth: float,
    chunk_size: int,
    min_samples: int,
    max_samples: int,
) -> torch.Tensor:
    return safe_mmd_loss(
        features,
        sensitive,
        bandwidth=bandwidth,
        chunk_size=chunk_size,
        min_samples=min_samples,
        max_samples=max_samples,
    )


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
    """Adapt the frozen source model to the target graph and return state v2."""

    ablation = str(getattr(args, "ablation", "full"))
    if ablation not in ABLATION_MODES:
        raise ValueError(
            "ablation must be one of {}; got {!r}".format(ABLATION_MODES, ablation)
        )
    prior_update_mode = str(getattr(args, "prior_update_mode", "ema"))
    if prior_update_mode not in PRIOR_UPDATE_MODES:
        raise ValueError(
            "prior_update_mode must be one of {}; got {!r}".format(
                PRIOR_UPDATE_MODES, prior_update_mode
            )
        )
    use_metaalign = ablation != "metaalign"
    use_bca = ablation != "bca"
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
    bottleneck = int(getattr(args, "adapter_bottleneck", max(1, dim // 4)))
    adapter = TargetAdapter(dim, bottleneck).to(device)

    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=float(getattr(args, "adapt_lr", 1e-3)),
    )

    source_prototypes = _initial_prototypes(knowledge, device)
    prototypes = source_prototypes.detach().clone()
    smoothing = float(getattr(args, "prior_count_smoothing", 1.0))
    soft_counts = _initial_soft_counts(knowledge, device, smoothing)
    initial_soft_counts = soft_counts.detach().clone()
    group_prior, global_prior = priors_from_counts(soft_counts)
    sensitive = target_data.sens_labels.to(device).long().view(-1)

    epochs = max(1, int(getattr(args, "adapt_epochs", 200)))
    keep_fraction = float(getattr(args, "tau_c", 0.2))
    temperature = float(getattr(args, "proto_temp", 0.5))
    configured_prior_weight = float(getattr(args, "lambda_pi", 1.0)) * float(
        getattr(args, "tau_adjust", 1.0)
    )
    prior_weight = configured_prior_weight if use_bca else 0.0
    alpha_p = min(max(float(getattr(args, "alpha_p", 0.9)), 0.0), 1.0)
    alpha_pi = min(max(float(getattr(args, "alpha_pi", 0.9)), 0.0), 1.0)
    lambda_cls = float(getattr(args, "lambda_s", 1.0))
    lambda_ent = float(getattr(args, "lambda_e", 0.1))
    lambda_anchor = float(getattr(args, "lambda_anchor", 1.0))
    lambda_fair = float(getattr(args, "target_lambda_fair", 1.0))
    meta_lr = float(getattr(args, "meta_lr", getattr(args, "meta_inner_lr", 0.01)))
    bandwidth = float(getattr(args, "mmd_bandwidth", 1.0))
    chunk_size = int(getattr(args, "mmd_chunk_size", 1024))
    min_mmd_samples = max(2, int(getattr(args, "target_mmd_min_samples", 2)))
    max_mmd_samples = max(0, int(getattr(args, "target_mmd_max_samples", 1024)))
    warmup_ratio = float(getattr(args, "fairness_warmup_ratio", 0.1))

    mmd_steps = 0
    last_losses = {}
    max_adapter_grad_norm = 0.0
    selected_count = 0
    prior_update_steps = 0

    print(
        "[FairMAC] ablation={} prior_update={} epochs={} adapter={} bottleneck={}".format(
            ablation, prior_update_mode, epochs, dim, bottleneck
        )
    )

    for step in range(epochs):
        adapter.train()
        student_features = adapter(base_embedding)
        with torch.no_grad():
            reference_features = student_features.detach()
            reference_probabilities = compute_posterior(
                reference_features,
                prototypes,
                group_prior,
                sensitive,
                temperature,
                prior_weight,
            )
            confidence, pseudo_labels = reference_probabilities.max(1)
            selected = select_high_confidence(
                confidence,
                keep_fraction,
            )
        selected_count = int(selected.sum().item())

        current_prototypes = build_prototypes(
            student_features,
            pseudo_labels,
            confidence,
            selected,
            sensitive,
            prototypes,
            use_sensitive_residual=use_sensitive_residual,
        )

        mmd_loss = compute_target_fairness_loss(
            student_features,
            sensitive,
            bandwidth,
            chunk_size,
            min_mmd_samples,
            max_mmd_samples,
        )
        mmd_steps += 1

        ramp = _fairness_weight(step, epochs, warmup_ratio)
        fairness_loss = ramp * lambda_fair * mmd_loss

        if use_metaalign:
            adapter_parameters = dict(adapter.named_parameters())
            fairness_gradients = torch.autograd.grad(
                fairness_loss,
                tuple(adapter_parameters.values()),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            virtual_parameters = {
                name: parameter - meta_lr * gradient
                if gradient is not None
                else parameter
                for (name, parameter), gradient in zip(
                    adapter_parameters.items(), fairness_gradients
                )
            }
            virtual_features = functional_call(
                adapter, virtual_parameters, (base_embedding,)
            )
            virtual_prototypes = build_prototypes(
                virtual_features,
                pseudo_labels,
                confidence,
                selected,
                sensitive,
                prototypes,
                use_sensitive_residual=use_sensitive_residual,
            )
        else:
            virtual_features = student_features
            virtual_prototypes = current_prototypes
        virtual_probabilities = compute_posterior(
            virtual_features,
            virtual_prototypes,
            group_prior,
            sensitive,
            temperature,
            prior_weight,
        ).clamp_min(1e-8)

        if selected.any():
            classification_loss = F.nll_loss(
                virtual_probabilities[selected].log(),
                pseudo_labels[selected],
            )
        else:
            classification_loss = _graph_zero(virtual_probabilities)
        entropy_loss = -(
            virtual_probabilities * virtual_probabilities.log()
        ).sum(1).mean()
        feature_anchor = F.mse_loss(student_features, base_embedding)
        prototype_anchor = F.mse_loss(current_prototypes, source_prototypes)
        anchor_loss = feature_anchor + 0.25 * prototype_anchor

        total_loss = (
            lambda_cls * classification_loss
            + lambda_ent * entropy_loss
            + fairness_loss
            + lambda_anchor * anchor_loss
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
            prototypes = _safe_normalize(
                alpha_p * prototypes + (1.0 - alpha_p) * current_prototypes.detach(),
                dim=1,
            )
            new_counts = estimate_soft_counts(
                reference_probabilities,
                sensitive,
                smoothing,
            )
            soft_counts = update_prior_counts(
                soft_counts,
                new_counts,
                prior_update_mode,
                alpha_pi,
                smoothing,
            )
            if prior_update_mode != "frozen":
                prior_update_steps += 1
            group_prior, global_prior = priors_from_counts(soft_counts)

        last_losses = {
            "total": float(total_loss.detach().item()),
            "classification": float(classification_loss.detach().item()),
            "entropy": float(entropy_loss.detach().item()),
            "anchor": float(anchor_loss.detach().item()),
            "mmd": float(mmd_loss.detach().item()),
        }
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == epochs:
            print(
                "  [Adapt {:3d}] hc={}/{} cls={:.4f} "
                "mmd={:.4f} anchor={:.4f} grad={:.4e}".format(
                    step,
                    selected_count,
                    base_embedding.shape[0],
                    last_losses["classification"],
                    last_losses["mmd"],
                    last_losses["anchor"],
                    grad_norm,
                )
            )

    # Recompute final prototypes from the final adapter without updating modules.
    adapter.eval()
    with torch.no_grad():
        final_features = adapter(base_embedding)
        reference_features = final_features
        reference_probabilities = compute_posterior(
            reference_features,
            prototypes,
            group_prior,
            sensitive,
            temperature,
            prior_weight,
        )
        final_confidence, final_pseudo = reference_probabilities.max(1)
        final_selected = select_high_confidence(
            final_confidence,
            keep_fraction,
        )
        candidate_prototypes = build_prototypes(
            final_features,
            final_pseudo,
            final_confidence,
            final_selected,
            sensitive,
            prototypes,
            use_sensitive_residual=use_sensitive_residual,
        )
        final_prototypes = _safe_normalize(
            alpha_p * prototypes + (1.0 - alpha_p) * candidate_prototypes,
            dim=1,
        )
        candidate_counts = estimate_soft_counts(
            reference_probabilities, sensitive, smoothing
        )
        preliminary_counts = update_prior_counts(
            soft_counts,
            candidate_counts,
            prior_update_mode,
            alpha_pi,
            smoothing,
        )
        preliminary_group_prior, _ = priors_from_counts(preliminary_counts)
        final_reference_probabilities = compute_posterior(
            reference_features,
            final_prototypes,
            preliminary_group_prior,
            sensitive,
            temperature,
            prior_weight,
        )
        refined_counts = estimate_soft_counts(
            final_reference_probabilities, sensitive, smoothing
        )
        final_counts = update_prior_counts(
            soft_counts,
            refined_counts,
            prior_update_mode,
            alpha_pi,
            smoothing,
        )
        if prior_update_mode != "frozen":
            prior_update_steps += 1
        group_prior, global_prior = priors_from_counts(final_counts)

    target_joint_prior = final_counts / final_counts.sum().clamp_min(1e-8)
    shared_target_residual = {
        (class_id, group_id): torch.zeros_like(
            final_prototypes[class_id], device="cpu"
        )
        for class_id in (0, 1)
        for group_id in (0, 1)
    }
    target_pi_ys = {
        (class_id, group_id): float(target_joint_prior[group_id, class_id].item())
        for class_id in (0, 1)
        for group_id in (0, 1)
    }

    state = {
        "state_version": STATE_VERSION,
        "ablation": ablation,
        "use_metaalign": use_metaalign,
        "use_bca": use_bca,
        "use_sensitive_residual": use_sensitive_residual,
        "prior_update_mode": prior_update_mode,
        "configured_alpha_pi": alpha_pi,
        "adapter_config": adapter.config(),
        "adapter_state_dict": _cpu_state_dict(adapter),
        "final_protos": {
            class_id: final_prototypes[class_id].detach().cpu().clone()
            for class_id in (0, 1)
        },
        "soft_counts": final_counts.detach().cpu().clone(),
        "group_prior": group_prior.detach().cpu().clone(),
        "global_prior": global_prior.detach().cpu().clone(),
        "inference_strategy": "group_conditioned",
        "proto_temp": temperature,
        "lambda_pi": float(getattr(args, "lambda_pi", 1.0)) if use_bca else 0.0,
        "tau_adjust": float(getattr(args, "tau_adjust", 1.0)),
        "diagnostics": {
            "mmd_steps": mmd_steps,
            "last_losses": last_losses,
            "adapter_grad_norm_max": max_adapter_grad_norm,
            "source_encoder_frozen": all(
                not parameter.requires_grad for parameter in source_model.parameters()
            ),
            "selected_count": int(final_selected.sum().item()),
            "prior_update_steps": prior_update_steps,
            "prior_shift": float(
                (
                    priors_from_counts(initial_soft_counts)[0] - group_prior
                ).abs().mean().item()
            ),
        },
        # Compatibility metadata for existing analysis code.  New inference
        # relies only on the explicit v2 fields above.
        "p_y": {
            class_id: final_prototypes[class_id].detach().cpu().clone()
            for class_id in (0, 1)
        },
        "r_ys": shared_target_residual,
        "pi_ys": target_pi_ys,
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
    sensitive: torch.Tensor,
    state: Mapping,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(state.get("state_version", 0)) < STATE_VERSION:
        raise ValueError("predict_from_adaptation_state requires a version-2 adaptation state")
    if state.get("inference_strategy", "group_conditioned") != "group_conditioned":
        raise ValueError(
            "The adaptation state uses an unsupported inference strategy"
        )
    device = base_embedding.device
    adapter = build_adapter_from_state(state, device)
    prototypes = torch.stack(
        [state["final_protos"][class_id].to(device) for class_id in (0, 1)],
        dim=0,
    )
    group_prior = state["group_prior"].to(device)
    features = adapter(_safe_normalize(base_embedding, dim=1))
    prior_weight = float(state.get("lambda_pi", 1.0)) * float(
        state.get("tau_adjust", 1.0)
    )
    probabilities = compute_posterior(
        features,
        prototypes,
        group_prior,
        sensitive.to(device),
        float(state.get("proto_temp", 1.0)),
        prior_weight,
    )
    return features, probabilities


def predict_target_proba(args, data, encoder, state: Mapping) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return FairMAC-adapted features and target class probabilities."""
    encoder.eval()
    with torch.no_grad():
        base_embedding, _ = encoder(data.x, data.edge_index)
        return predict_from_adaptation_state(
            base_embedding,
            data.sens_labels,
            state,
        )


def save_adaptation_state(state: Mapping, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(dict(state), path)


def load_adaptation_state(path: str, map_location="cpu") -> Dict:
    return torch.load(path, map_location=map_location)
