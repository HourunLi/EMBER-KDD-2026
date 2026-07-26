"""Paper-aligned source-free target adaptation for EMBER.

The source model is frozen on the target graph.  Target adaptation updates
only class prototype residuals and discounted soft-evidence class priors.
Target labels and sensitive attributes are never consumed by this module.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_IDS = (0, 1)
GROUP_IDS = (0, 1)
JOINT_KEYS = tuple(
    (class_id, group_id)
    for class_id in CLASS_IDS
    for group_id in GROUP_IDS
)
PAPER_ABLATION_MODES = (
    "decouple",
    "metaalign",
    "minority",
    "bca",
    "groupinit",
)
# Keep the two former modes callable for old tuning scripts and checkpoints,
# but do not include them in the paper's five-ablation runner.
LEGACY_ABLATION_MODES = ("ema", "residual")
ABLATION_MODES = ("full", *PAPER_ABLATION_MODES, *LEGACY_ABLATION_MODES)


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x), p=2, dim=dim, eps=1e-12)


def _graph_zero(x: torch.Tensor) -> torch.Tensor:
    """Return a zero connected to ``x`` when ``x`` requires gradients."""
    return x.sum() * 0.0


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_positive(name: str, value: float, allow_zero: bool = False) -> float:
    value = _require_finite(name, value)
    invalid = value < 0.0 if allow_zero else value <= 0.0
    if invalid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return value


def _require_unit_interval(
    name: str,
    value: float,
    include_one: bool = True,
) -> float:
    value = _require_finite(name, value)
    valid = 0.0 <= value <= 1.0 if include_one else 0.0 <= value < 1.0
    if not valid:
        interval = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return value


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

    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    cap = int(max_samples)
    if cap > 0:
        if h0.shape[0] > cap:
            indices = torch.randperm(h0.shape[0], device=h0.device)[:cap]
            h0 = h0[indices]
        if h1.shape[0] > cap:
            indices = torch.randperm(h1.shape[0], device=h1.device)[:cap]
            h1 = h1[indices]

    sigma = _require_positive("MMD bandwidth", bandwidth)
    block = int(chunk_size)

    def kernel_mean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = _graph_zero(x)
        count = 0
        denominator = 2.0 * sigma * sigma
        for i in range(0, x.shape[0], block):
            xb = x[i : i + block]
            x2 = xb.square().sum(1, keepdim=True)
            for j in range(0, y.shape[0], block):
                yb = y[j : j + block]
                y2 = yb.square().sum(1).unsqueeze(0)
                distance2 = (x2 + y2 - 2.0 * xb @ yb.t()).clamp_min(0.0)
                total = total + torch.exp(-distance2 / denominator).sum()
                count += xb.shape[0] * yb.shape[0]
        return total / max(count, 1)

    value = (
        kernel_mean(h0, h0)
        + kernel_mean(h1, h1)
        - 2.0 * kernel_mean(h0, h1)
    )
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def class_conditional_mmd_loss(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    group_labels: torch.Tensor,
    selected: torch.Tensor,
    bandwidth: float,
    chunk_size: int,
    min_samples: int,
    max_samples: int,
    reduction: str = "sum",
) -> Tuple[torch.Tensor, int]:
    """Shared class-conditional MMD used by Source and Target training."""
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")

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
    if reduction == "mean":
        # Eq. (6) divides by the task label-space size |Y|, not by the number
        # of classes that happened to contain enough samples in this batch.
        loss = loss / float(len(CLASS_IDS))
    return loss, valid_classes


def source_classifier_probabilities(classifier_logits: torch.Tensor) -> torch.Tensor:
    """Convert the frozen Source classifier output to class probabilities."""
    if classifier_logits.ndim == 1:
        classifier_logits = classifier_logits.unsqueeze(1)
    if classifier_logits.shape[1] == 1:
        positive = torch.sigmoid(classifier_logits[:, 0])
        return torch.stack((1.0 - positive, positive), dim=1)
    return torch.softmax(classifier_logits, dim=1)


def select_high_confidence(
    confidence: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Eqs. (12), (17), and (20): threshold a confidence score."""
    delta = _require_unit_interval("confidence threshold", threshold)
    return confidence > delta


def _source_bank(
    knowledge: Mapping,
    bank_name: str,
    indices: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    bank = knowledge.get(bank_name)
    if not isinstance(bank, Mapping):
        raise ValueError(f"Source knowledge does not contain '{bank_name}'")

    values = []
    feature_dim = None
    for index in indices:
        value = bank.get(index)
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(f"{bank_name}[{index}] must be a one-dimensional tensor")
        if not bool(torch.isfinite(value).all()) or value.norm().item() <= 1e-12:
            raise ValueError(f"{bank_name}[{index}] must be finite and non-zero")
        if feature_dim is None:
            feature_dim = value.numel()
        elif value.numel() != feature_dim:
            raise ValueError(f"All vectors in '{bank_name}' must have equal size")
        values.append(_safe_normalize(value.to(device).float(), dim=0))
    return torch.stack(values, dim=0)


def _source_class_prototypes(
    knowledge: Mapping,
    device: torch.device,
) -> torch.Tensor:
    return _source_bank(knowledge, "p_y", CLASS_IDS, device)


def _source_group_anchors(
    knowledge: Mapping,
    device: torch.device,
) -> torch.Tensor:
    return _source_bank(knowledge, "a_g", GROUP_IDS, device)


def infer_pseudo_sensitive(
    group_features: torch.Tensor,
    source_group_anchors: torch.Tensor,
) -> torch.Tensor:
    """Eq. (12): match each group-recovery view to its nearest source anchor."""
    similarities = (
        _safe_normalize(group_features, dim=1)
        @ _safe_normalize(source_group_anchors, dim=1).t()
    )
    return similarities.argmax(dim=1)


def prototype_logits(
    features: torch.Tensor,
    class_prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = _require_positive("prototype temperature", temperature)
    return (
        _safe_normalize(features, dim=1)
        @ _safe_normalize(class_prototypes, dim=1).t()
    ) / temperature


def prototype_probabilities(
    features: torch.Tensor,
    class_prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return torch.softmax(
        prototype_logits(features, class_prototypes, temperature), dim=1
    )


def bayesian_class_posterior(
    features: torch.Tensor,
    class_prototypes: torch.Tensor,
    class_prior: torch.Tensor,
    temperature: float,
    prior_strength: float,
) -> torch.Tensor:
    """Eq. (18): prototype likelihood plus tempered target class prior."""
    prior_strength = _require_unit_interval("Bayesian prior strength", prior_strength)
    corrected_logits = prototype_logits(features, class_prototypes, temperature)
    corrected_logits = corrected_logits + prior_strength * torch.log(
        class_prior.clamp_min(1e-8)
    ).unsqueeze(0)
    return torch.softmax(torch.nan_to_num(corrected_logits), dim=1)


def minority_aware_weights(
    pseudo_labels: torch.Tensor,
    pseudo_sensitive: torch.Tensor,
    selected: torch.Tensor,
    pseudocount: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. (14): inverse pseudo-group count with Dirichlet smoothing."""
    pseudocount = _require_positive("group pseudocount", pseudocount)
    counts = torch.zeros(
        (len(CLASS_IDS), len(GROUP_IDS)),
        device=pseudo_labels.device,
    )
    for class_id, group_id in JOINT_KEYS:
        counts[class_id, group_id] = (
            selected
            & (pseudo_labels == class_id)
            & (pseudo_sensitive == group_id)
        ).sum()

    weights = torch.zeros_like(pseudo_labels, dtype=torch.float)
    if selected.any():
        selected_classes = pseudo_labels[selected].long()
        selected_groups = pseudo_sensitive[selected].long()
        weights[selected] = 1.0 / (
            counts[selected_classes, selected_groups] + pseudocount
        )
    return weights, counts


def _prior_from_evidence(
    evidence: torch.Tensor,
    pseudocount: float,
) -> torch.Tensor:
    """Eq. (17): posterior mean of the symmetric Dirichlet prior."""
    n0 = _require_positive("prior pseudocount", pseudocount)
    if not bool(torch.isfinite(evidence).all()) or bool((evidence < 0).any()):
        raise ValueError("Class-prior evidence must be finite and non-negative")
    numerator = evidence + n0
    denominator = len(CLASS_IDS) * n0 + evidence.sum()
    return numerator / denominator


def update_class_prior(
    accumulated_evidence: torch.Tensor,
    round_evidence: torch.Tensor,
    pseudocount: float,
    discount: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. (17): discount past evidence and return the smoothed prior.

    ``discount=0`` removes the historical-evidence update and re-estimates
    the target prior from the current round.  The legacy ``ema`` mode exposes
    this prior-history ablation for compatibility; it is not the Eq. (16)
    prototype-average ablation and is not part of the five paper variants.
    """
    if accumulated_evidence.shape != round_evidence.shape:
        raise ValueError(
            "Accumulated and current-round class-prior evidence must have "
            "the same shape"
        )
    if not bool(torch.isfinite(accumulated_evidence).all()) or bool(
        (accumulated_evidence < 0).any()
    ):
        raise ValueError(
            "Accumulated class-prior evidence must be finite and non-negative"
        )
    if not bool(torch.isfinite(round_evidence).all()) or bool(
        (round_evidence < 0).any()
    ):
        raise ValueError(
            "Current-round class-prior evidence must be finite and non-negative"
        )

    omega = _require_unit_interval(
        "omega",
        discount,
        include_one=False,
    )
    updated_evidence = omega * accumulated_evidence + round_evidence
    return updated_evidence, _prior_from_evidence(updated_evidence, pseudocount)


def _base_state(
    args,
    class_prototypes: torch.Tensor,
    class_prior: torch.Tensor,
    prediction_mode: str = "prototype_posterior",
) -> Dict:
    ablation = str(getattr(args, "ablation", "full"))
    use_bca = ablation != "bca"
    return {
        "class_prototypes": class_prototypes.detach().cpu().clone(),
        "class_prior": class_prior.detach().cpu().clone(),
        "proto_temp": float(getattr(args, "proto_temp", 1.0)),
        "prior_strength": (
            float(getattr(args, "lambda_pi", 1.0)) if use_bca else 0.0
        ),
        "prediction_mode": prediction_mode,
    }


def build_initial_adaptation_state(args, knowledge: Mapping) -> Dict:
    """Build the frozen-source baseline state without Target annotations."""
    device = args.device
    class_prototypes = _source_class_prototypes(knowledge, device)
    class_prior = _prior_from_evidence(
        torch.zeros(len(CLASS_IDS), device=device),
        getattr(args, "prior_pseudocount", 1.0),
    )
    return _base_state(
        args,
        class_prototypes,
        class_prior,
        prediction_mode="source_classifier",
    )


def run_target_adaptation(
    args,
    source_model: nn.Module,
    target_data,
    knowledge: Mapping,
) -> Dict:
    """Run Eqs. (12)-(20) while keeping the source model frozen."""
    ablation = str(getattr(args, "ablation", "full"))
    if ablation not in ABLATION_MODES:
        raise ValueError(
            "ablation must be one of {}; got {!r}".format(ABLATION_MODES, ablation)
        )

    use_bca = ablation != "bca"
    use_prior_history = ablation != "ema"
    use_residual = ablation != "residual"
    use_minority_weights = ablation != "minority"
    anchor_space = "task" if ablation == "decouple" else "sensitive"

    device = args.device
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    with torch.no_grad():
        task_features, sensitive_features = source_model.encode_views(
            target_data.x,
            target_data.edge_index,
        )
        source_logits = source_model.cls_head(task_features)
        features = _safe_normalize(task_features.detach(), dim=1)
        sensitive_features = _safe_normalize(
            sensitive_features.detach(), dim=1
        )
        source_probabilities = source_classifier_probabilities(source_logits.detach())

    source_class_prototypes = _source_class_prototypes(knowledge, device)
    knowledge_anchor_space = str(knowledge.get("anchor_space", "sensitive"))
    if knowledge_anchor_space != anchor_space:
        raise ValueError(
            "Source anchor space mismatch: ablation={!r} requires {!r} anchors, "
            "but the knowledge package contains {!r} anchors".format(
                ablation,
                anchor_space,
                knowledge_anchor_space,
            )
        )
    source_group_anchors = _source_group_anchors(knowledge, device)
    class_prototypes = source_class_prototypes.detach().clone()

    discounted_evidence = torch.zeros(len(CLASS_IDS), device=device)
    class_prior = _prior_from_evidence(
        discounted_evidence,
        getattr(args, "prior_pseudocount", 1.0),
    )
    class_update_counts = torch.zeros(
        len(CLASS_IDS),
        device=device,
        dtype=torch.long,
    )

    epochs = int(getattr(args, "adapt_epochs", 200))
    residual_inner_steps = int(getattr(args, "residual_inner_steps", 5))
    if epochs < 1:
        raise ValueError("adapt_epochs must be at least 1")
    if residual_inner_steps < 1:
        raise ValueError("residual_inner_steps must be at least 1")
    adapt_lr = _require_positive(
        "adaptation learning rate",
        getattr(args, "adapt_lr", 1e-3),
    )
    threshold = _require_unit_interval("delta", getattr(args, "tau_c", 0.7))
    prior_threshold = _require_unit_interval(
        "delta_p",
        getattr(args, "prior_confidence_threshold", threshold),
    )
    temperature = _require_positive(
        "prototype temperature",
        getattr(args, "proto_temp", 1.0),
    )
    configured_prior_strength = _require_unit_interval(
        "eta",
        getattr(args, "lambda_pi", 1.0),
    )
    prior_strength = configured_prior_strength if use_bca else 0.0
    residual_l2_weight = _require_positive(
        "residual L2 weight",
        getattr(args, "lambda_residual_l2", 1e-3),
        allow_zero=True,
    )
    group_pseudocount = _require_positive(
        "group pseudocount",
        getattr(args, "group_pseudocount", 1.0),
    )
    prior_pseudocount = _require_positive(
        "prior pseudocount",
        getattr(args, "prior_pseudocount", 1.0),
    )
    prior_discount = _require_unit_interval(
        "omega",
        getattr(args, "prior_discount", 0.9),
        include_one=False,
    )
    effective_prior_discount = prior_discount if use_prior_history else 0.0

    # Eq. (12): task confidence is read from the frozen source classifier.
    # The full method recovers groups in the decoupled sensitive space; the
    # decouple ablation deliberately uses the fairness-aligned task space.
    source_confidence, pseudo_labels = source_probabilities.max(dim=1)
    source_confidence = source_confidence.detach()
    group_features = features if ablation == "decouple" else sensitive_features
    pseudo_sensitive = infer_pseudo_sensitive(
        group_features,
        source_group_anchors,
    )
    selected = select_high_confidence(source_confidence, threshold)

    print(
        "[EMBER] ablation={} rounds={} residual_inner={} "
        "residual={} group_weights={} anchors={} prototype_average=True "
        "prior={} prior_history={}".format(
            ablation,
            epochs,
            residual_inner_steps,
            use_residual,
            "inverse-frequency" if use_minority_weights else "uniform",
            anchor_space,
            "count-aware" if use_bca else "disabled",
            "discounted" if use_prior_history else "current-round-only",
        )
    )

    for step in range(epochs):
        residual_nll = _graph_zero(class_prototypes)
        residual_l2 = _graph_zero(class_prototypes)
        if use_residual:
            if use_minority_weights:
                group_weights, _ = minority_aware_weights(
                    pseudo_labels,
                    pseudo_sensitive,
                    selected,
                    group_pseudocount,
                )
            else:
                # Ablate Eq. (14) only: confidence weighting, residual
                # optimization and cumulative prototype averaging all remain.
                group_weights = torch.ones_like(source_confidence)
            step_residuals = nn.Parameter(torch.zeros_like(class_prototypes))
            step_optimizer = torch.optim.Adam([step_residuals], lr=adapt_lr)
            node_weights = source_confidence * group_weights
            normalizer = node_weights[selected].sum()

            for _ in range(residual_inner_steps):
                candidate_prototypes = _safe_normalize(
                    class_prototypes.detach() + step_residuals,
                    dim=1,
                )
                log_prototype_probabilities = F.log_softmax(
                    prototype_logits(features, candidate_prototypes, temperature),
                    dim=1,
                )

                if selected.any() and normalizer.detach().item() > 1e-12:
                    row_index = torch.arange(features.shape[0], device=device)
                    point_losses = -log_prototype_probabilities[
                        row_index,
                        pseudo_labels.long(),
                    ]
                    residual_nll = (
                        node_weights[selected] * point_losses[selected]
                    ).sum() / normalizer.clamp_min(1e-8)
                else:
                    residual_nll = _graph_zero(candidate_prototypes)

                residual_l2 = step_residuals.square().sum()
                total_loss = residual_nll + residual_l2_weight * residual_l2
                step_optimizer.zero_grad()
                total_loss.backward()
                step_optimizer.step()

            with torch.no_grad():
                optimized_candidates = _safe_normalize(
                    class_prototypes + step_residuals,
                    dim=1,
                )
                updated_prototypes = class_prototypes.clone()
                for class_id in CLASS_IDS:
                    class_selected = selected & (pseudo_labels == class_id)
                    if not class_selected.any():
                        continue
                    # Eq. (16) remains active for all five paper ablations.
                    class_update_counts[class_id] += 1
                    mu = 1.0 / float(
                        class_update_counts[class_id].item() + 1
                    )
                    updated_prototypes[class_id] = _safe_normalize(
                        (1.0 - mu) * class_prototypes[class_id]
                        + mu * optimized_candidates[class_id],
                        dim=0,
                    )
                class_prototypes = updated_prototypes

        with torch.no_grad():
            likelihood = prototype_probabilities(
                features,
                class_prototypes,
                temperature,
            )
            if use_bca:
                prior_selected = select_high_confidence(
                    likelihood.max(dim=1).values,
                    prior_threshold,
                )
                round_evidence = (
                    source_confidence[prior_selected].unsqueeze(1)
                    * likelihood[prior_selected]
                ).sum(dim=0)
                discounted_evidence, class_prior = update_class_prior(
                    discounted_evidence,
                    round_evidence,
                    prior_pseudocount,
                    effective_prior_discount,
                )
                adapted_probabilities = bayesian_class_posterior(
                    features,
                    class_prototypes,
                    class_prior,
                    temperature,
                    prior_strength,
                )
            else:
                adapted_probabilities = likelihood
            adapted_confidence, pseudo_labels = adapted_probabilities.max(dim=1)
            selected = select_high_confidence(adapted_confidence, threshold)

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == epochs:
            print(
                "  [Adapt {:3d}] hc={}/{} res={:.4f} l2={:.4f} "
                "prior=({:.3f},{:.3f})".format(
                    step + 1,
                    int(selected.sum().item()),
                    features.shape[0],
                    float(residual_nll.detach().item()),
                    float(residual_l2.detach().item()),
                    float(class_prior[0].item()),
                    float(class_prior[1].item()),
                )
            )

    return _base_state(args, class_prototypes, class_prior)


def predict_target_proba(
    args,
    data,
    encoder,
    state: Mapping,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return frozen-source or Eq. (18) probabilities without annotations."""
    del args
    encoder.eval()
    with torch.no_grad():
        task_features, source_logits = encoder(data.x, data.edge_index)
        device = task_features.device
        features = _safe_normalize(task_features, dim=1)
        if state.get("prediction_mode") == "source_classifier":
            probabilities = source_classifier_probabilities(source_logits)
        else:
            probabilities = bayesian_class_posterior(
                features,
                state["class_prototypes"].to(device),
                state["class_prior"].to(device),
                float(state["proto_temp"]),
                float(state["prior_strength"]),
            )
        return features, probabilities
