"""Paper-aligned source-free target adaptation for FairMAC.

The source model is frozen on the target graph.  Target adaptation updates
only class prototype residuals and confidence-counted target class priors.
Target labels and sensitive attributes are never consumed by this module.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


JOINT_KEYS = ((0, 0), (0, 1), (1, 0), (1, 1))
ABLATION_MODES = ("full", "metaalign", "bca", "target_mmd", "residual")


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(torch.nan_to_num(x), p=2, dim=dim, eps=1e-12)


def _graph_zero(x: torch.Tensor) -> torch.Tensor:
    """Return a zero connected to ``x`` when ``x`` requires gradients."""
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
            indices = torch.randperm(h0.shape[0], device=h0.device)[:cap]
            h0 = h0[indices]
        if h1.shape[0] > cap:
            indices = torch.randperm(h1.shape[0], device=h1.device)[:cap]
            h1 = h1[indices]

    sigma = max(float(bandwidth), 1e-6)
    block = max(int(chunk_size), 1)

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
    if reduction == "mean" and selected_classes:
        loss = loss / float(len(selected_classes))
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
    """Eq. (8)/(14): select nodes whose confidence is above delta."""
    delta = min(max(float(threshold), 0.0), 1.0)
    return confidence > delta


def _source_class_prototypes(
    knowledge: Mapping,
    device: torch.device,
) -> torch.Tensor:
    values = []
    dim = int(knowledge["p_y"][0].numel())
    for class_id in (0, 1):
        value = knowledge["p_y"].get(class_id)
        if value is None or not torch.isfinite(value).all() or value.norm() < 1e-12:
            value = torch.zeros(dim)
            value[class_id % dim] = 1.0
        values.append(_safe_normalize(value.to(device).float(), dim=0))
    return torch.stack(values, dim=0)


def _source_group_prototypes(
    knowledge: Mapping,
    device: torch.device,
) -> torch.Tensor:
    if "p_ys" not in knowledge:
        raise ValueError(
            "Source knowledge does not contain group prototypes 'p_ys'; "
            "regenerate the Source checkpoint with the group-balanced exporter"
        )
    dim = int(knowledge["p_y"][0].numel())
    prototypes = torch.zeros((2, 2, dim), device=device)
    for class_id, group_id in JOINT_KEYS:
        value = knowledge["p_ys"].get((class_id, group_id))
        if value is None or not torch.isfinite(value).all() or value.norm() < 1e-12:
            value = knowledge["p_y"][class_id]
        prototypes[class_id, group_id] = _safe_normalize(
            value.to(device).float(), dim=0
        )
    return prototypes


def infer_pseudo_sensitive(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    source_group_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Eq. (8)/(15): nearest Source group prototype inside pseudo class."""
    class_group_prototypes = source_group_prototypes[pseudo_labels.long()]
    similarities = torch.einsum(
        "nd,ngd->ng",
        _safe_normalize(features, dim=1),
        _safe_normalize(class_group_prototypes, dim=2),
    )
    return similarities.argmax(dim=1)


def prototype_logits(
    features: torch.Tensor,
    class_prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return (
        _safe_normalize(features, dim=1)
        @ _safe_normalize(class_prototypes, dim=1).t()
    ) / max(float(temperature), 1e-6)


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
    """Eq. (12)/(17): prototype likelihood plus count-aware class prior."""
    corrected_logits = prototype_logits(features, class_prototypes, temperature)
    corrected_logits = corrected_logits + float(prior_strength) * torch.log(
        class_prior.clamp_min(1e-8)
    ).unsqueeze(0)
    return torch.softmax(torch.nan_to_num(corrected_logits), dim=1)


def minority_aware_weights(
    pseudo_labels: torch.Tensor,
    pseudo_sensitive: torch.Tensor,
    selected: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. (9): inverse class-conditional pseudo-group frequency weights."""
    counts = torch.zeros((2, 2), device=pseudo_labels.device)
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
            counts[selected_classes, selected_groups]
            + max(float(epsilon), 1e-12)
        )
    return weights, counts


def class_centered_features(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    candidate_prototypes: torch.Tensor,
) -> torch.Tensor:
    """Residual-dependent Target representation used by the added MMD term."""
    centers = candidate_prototypes[pseudo_labels.long()]
    return _safe_normalize(features - centers, dim=1)


def _initial_count_prior(
    pseudocount: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = torch.full(
        (2,), max(float(pseudocount), 0.0), device=device, dtype=torch.float
    )
    if counts.sum().item() <= 1e-12:
        counts.fill_(1.0)
    prior = (counts + 1e-8) / (counts + 1e-8).sum().clamp_min(1e-8)
    return counts, prior


def _base_state(
    args,
    class_prototypes: torch.Tensor,
    class_prior: torch.Tensor,
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
    }


def build_initial_adaptation_state(args, knowledge: Mapping) -> Dict:
    """Build the t=0 Source-prototype state without touching Target labels."""
    device = args.device
    class_prototypes = _source_class_prototypes(knowledge, device)
    _, class_prior = _initial_count_prior(
        getattr(args, "prior_pseudocount", 1.0), device
    )
    return _base_state(args, class_prototypes, class_prior)


def run_target_adaptation(
    args,
    source_model: nn.Module,
    target_data,
    knowledge: Mapping,
) -> Dict:
    """Run Eqs. (8)-(17) while keeping the Source model completely frozen."""
    ablation = str(getattr(args, "ablation", "full"))
    if ablation not in ABLATION_MODES:
        raise ValueError(
            "ablation must be one of {}; got {!r}".format(ABLATION_MODES, ablation)
        )

    use_bca = ablation != "bca"
    use_residual = ablation != "residual"
    use_target_mmd = ablation != "target_mmd" and use_residual

    device = args.device
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    with torch.no_grad():
        base_embedding, source_logits = source_model(
            target_data.x, target_data.edge_index
        )
        features = _safe_normalize(base_embedding.detach(), dim=1)
        source_probabilities = source_classifier_probabilities(source_logits.detach())

    source_class_prototypes = _source_class_prototypes(knowledge, device)
    source_group_prototypes = _source_group_prototypes(knowledge, device)
    class_prototypes = source_class_prototypes.detach().clone()

    class_counts, class_prior = _initial_count_prior(
        getattr(args, "prior_pseudocount", 1.0), device
    )
    class_update_counts = torch.zeros(2, device=device, dtype=torch.long)

    epochs = max(1, int(getattr(args, "adapt_epochs", 200)))
    residual_inner_steps = max(
        1, int(getattr(args, "residual_inner_steps", 5))
    )
    adapt_lr = float(getattr(args, "adapt_lr", 1e-3))
    threshold = float(getattr(args, "tau_c", 0.7))
    temperature = float(getattr(args, "proto_temp", 1.0))
    prior_strength = float(getattr(args, "lambda_pi", 1.0)) if use_bca else 0.0
    residual_l2_weight = float(getattr(args, "lambda_residual_l2", 1e-3))
    minority_epsilon = float(getattr(args, "minority_epsilon", 1e-6))
    target_mmd_weight = float(getattr(args, "target_lambda_fair", 1.0))
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

    # Eq. (8): q_v^ta is produced once by the frozen Source classifier and
    # remains fixed throughout adaptation.  Bayesian confidence is introduced
    # separately by Eqs. (13)-(14) only to refresh subsequent confident sets.
    source_confidence, pseudo_labels = source_probabilities.max(dim=1)
    source_confidence = source_confidence.detach()
    selected = select_high_confidence(source_confidence, threshold)

    print(
        "[FairMAC] ablation={} epochs={} residual_inner={} "
        "trainable=class_residual prior=count-aware target_mmd={}".format(
            ablation, epochs, residual_inner_steps, use_target_mmd
        )
    )

    for step in range(epochs):
        with torch.no_grad():
            pseudo_sensitive = infer_pseudo_sensitive(
                features, pseudo_labels, source_group_prototypes
            )

        inverse_group_weights, _ = minority_aware_weights(
            pseudo_labels,
            pseudo_sensitive,
            selected,
            minority_epsilon,
        )

        # Eq. (9)/(16): every adaptation step owns an independent R^(t),
        # initialized at zero and discarded after updating the prototypes.
        step_residuals = nn.Parameter(
            torch.zeros_like(class_prototypes), requires_grad=use_residual
        )
        step_optimizer = (
            torch.optim.Adam([step_residuals], lr=adapt_lr)
            if use_residual
            else None
        )
        optimization_steps = residual_inner_steps if use_residual else 1
        # Eq. (9): q_v^ta is the fixed Source-classifier confidence, while
        # only the inverse pseudo-group-frequency term changes with t.
        node_weights = source_confidence * inverse_group_weights
        normalizer = node_weights[selected].sum()

        for _ in range(optimization_steps):
            candidate_prototypes = _safe_normalize(
                class_prototypes.detach() + step_residuals, dim=1
            )
            log_prototype_probabilities = F.log_softmax(
                prototype_logits(features, candidate_prototypes, temperature),
                dim=1,
            )

            if selected.any() and normalizer.detach().item() > 1e-12:
                row_index = torch.arange(features.shape[0], device=device)
                point_losses = -log_prototype_probabilities[
                    row_index, pseudo_labels.long()
                ]
                residual_nll = (
                    node_weights[selected] * point_losses[selected]
                ).sum() / normalizer.clamp_min(1e-8)
            else:
                residual_nll = _graph_zero(candidate_prototypes)

            residual_l2 = step_residuals.square().sum()

            if use_target_mmd:
                centered_features = class_centered_features(
                    features, pseudo_labels, candidate_prototypes
                )
                target_mmd, valid_mmd_classes = class_conditional_mmd_loss(
                    centered_features,
                    pseudo_labels,
                    pseudo_sensitive,
                    selected,
                    bandwidth=bandwidth,
                    chunk_size=chunk_size,
                    min_samples=min_mmd_samples,
                    max_samples=max_mmd_samples,
                    reduction="sum",
                )
            else:
                target_mmd = _graph_zero(candidate_prototypes)
                valid_mmd_classes = 0

            total_loss = (
                residual_nll
                + residual_l2_weight * residual_l2
                + target_mmd_weight * target_mmd
            )

            if step_optimizer is not None:
                step_optimizer.zero_grad()
                total_loss.backward()
                step_optimizer.step()

        with torch.no_grad():
            optimized_candidates = _safe_normalize(
                class_prototypes + step_residuals, dim=1
            )
            updated_prototypes = class_prototypes.clone()
            for class_id in (0, 1):
                class_selected = selected & (pseudo_labels == class_id)
                if not class_selected.any() or not use_residual:
                    continue
                class_update_counts[class_id] += 1
                mu = 1.0 / float(class_update_counts[class_id].item() + 1)
                updated_prototypes[class_id] = _safe_normalize(
                    (1.0 - mu) * class_prototypes[class_id]
                    + mu * optimized_candidates[class_id],
                    dim=0,
                )
            class_prototypes = updated_prototypes

            # Eq. (11): cumulative counts always use the fixed q_v^ta from
            # Eq. (8).  The prototype-induced label and confident set remain
            # iteration-dependent.
            likelihood = prototype_probabilities(
                features, class_prototypes, temperature
            )
            count_labels = likelihood.argmax(dim=1)
            for class_id in (0, 1):
                class_count_mask = selected & (count_labels == class_id)
                class_counts[class_id] += source_confidence[
                    class_count_mask
                ].sum()
            class_prior = (class_counts + 1e-8) / (
                class_counts + 1e-8
            ).sum().clamp_min(1e-8)

            # Eqs. (12)-(14): Bayesian confidence refreshes only the next
            # pseudo-labels and confident set; it never replaces q_v^ta in
            # either the residual objective or cumulative counting.
            corrected_probabilities = bayesian_class_posterior(
                features,
                class_prototypes,
                class_prior,
                temperature,
                prior_strength,
            )
            bayesian_confidence, pseudo_labels = corrected_probabilities.max(dim=1)
            selected = select_high_confidence(bayesian_confidence, threshold)

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == epochs:
            print(
                "  [Adapt {:3d}] hc={}/{} res={:.4f} l2={:.4f} "
                "mmd={:.4f} mmd_cls={} prior=({:.3f},{:.3f})".format(
                    step,
                    int(selected.sum().item()),
                    features.shape[0],
                    float(residual_nll.detach().item()),
                    float(residual_l2.detach().item()),
                    float(target_mmd.detach().item()),
                    valid_mmd_classes,
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
    """Eq. (17) inference without accessing Target labels or attributes."""
    del args
    encoder.eval()
    with torch.no_grad():
        base_embedding, _ = encoder(data.x, data.edge_index)
        device = base_embedding.device
        features = _safe_normalize(base_embedding, dim=1)
        probabilities = bayesian_class_posterior(
            features,
            state["class_prototypes"].to(device),
            state["class_prior"].to(device),
            float(state["proto_temp"]),
            float(state["prior_strength"]),
        )
        return features, probabilities
