"""Evaluation metrics shared by the NIFTY entry points.

The implementation follows ``learn(1).py``: utility metrics are macro-
averaged over the target classes, while demographic parity and equal
opportunity are absolute gaps between the two sensitive groups.
"""

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


MetricBySplit = Dict[str, float]


def _as_1d_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values).squeeze()
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional after squeezing.")
    return array


def _selection_to_indices(
    selection: Sequence[int],
    sample_count: int,
    split_name: str,
) -> np.ndarray:
    selection_array = np.asarray(selection)
    if selection_array.dtype == np.bool_:
        if selection_array.ndim != 1 or selection_array.size != sample_count:
            raise ValueError(
                f"Boolean selection for split {split_name!r} must have "
                f"length {sample_count}."
            )
        indices = np.flatnonzero(selection_array)
    else:
        indices = selection_array.astype(np.int64, copy=False).reshape(-1)

    if indices.size == 0:
        raise ValueError(f"Split {split_name!r} is empty.")
    if indices.min() < 0 or indices.max() >= sample_count:
        raise IndexError(f"Split {split_name!r} contains an out-of-range index.")
    return indices


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Compute sigmoid without overflowing for large-magnitude logits."""
    probabilities = np.empty_like(logits, dtype=np.float64)
    nonnegative = logits >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = exp_logits / (1.0 + exp_logits)
    return probabilities


def _absolute_rate_gap(
    predictions: np.ndarray,
    first_group: np.ndarray,
    second_group: np.ndarray,
) -> float:
    """Return an absolute positive-rate gap, or NaN for an absent group."""
    if not first_group.any() or not second_group.any():
        return float("nan")
    return float(abs(predictions[first_group].mean() - predictions[second_group].mean()))


def evaluate_per_class(
    labels: np.ndarray,
    logits: np.ndarray,
    sensitive: np.ndarray,
    splits: Mapping[str, Sequence[int]],
    *,
    percentage: bool = True,
) -> Tuple[MetricBySplit, MetricBySplit, MetricBySplit, MetricBySplit]:
    """Evaluate binary predictions using the method from ``learn(1).py``.

    Accuracy is the macro mean of the per-target-class accuracies. For binary
    labels this is balanced accuracy: ``(TPR + TNR) / 2``. AUC-ROC is computed
    one-vs-rest for each target class and macro-averaged. In the binary case,
    the class-0 AUC based on ``1 - probability`` equals the class-1 AUC.

    DP is the absolute difference in positive prediction rates between
    sensitive groups 0 and 1. EO is the same difference restricted to nodes
    whose true target label is 1. A missing class makes AUC undefined; a
    missing sensitive subgroup makes the corresponding fairness metric
    undefined. These cases are represented by ``NaN``.
    """
    labels_array = _as_1d_array(labels, "labels")
    logits_array = _as_1d_array(logits, "logits").astype(np.float64, copy=False)
    sensitive_array = _as_1d_array(sensitive, "sensitive")

    sample_count = labels_array.size
    if logits_array.size != sample_count or sensitive_array.size != sample_count:
        raise ValueError("labels, logits, and sensitive must have the same length.")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("labels must contain only binary values 0 and 1.")
    if not np.isin(sensitive_array, (0, 1)).all():
        raise ValueError("sensitive must contain only binary values 0 and 1.")
    labels_array = labels_array.astype(np.int64, copy=False)
    sensitive_array = sensitive_array.astype(np.int64, copy=False)

    probabilities = _sigmoid(logits_array)
    predictions = (probabilities > 0.5).astype(np.int64)
    scale = 100.0 if percentage else 1.0

    accuracies: MetricBySplit = {}
    auc_rocs: MetricBySplit = {}
    demographic_parities: MetricBySplit = {}
    equal_opportunities: MetricBySplit = {}

    for split_name, selection in splits.items():
        indices = _selection_to_indices(selection, sample_count, split_name)
        y_true = labels_array[indices]
        y_pred = predictions[indices]
        probability = probabilities[indices]
        split_sensitive = sensitive_array[indices]

        per_class_accuracy = []
        per_class_auc = []
        has_both_classes = np.unique(y_true).size == 2
        for target_class in np.unique(y_true):
            class_mask = y_true == target_class
            per_class_accuracy.append(
                accuracy_score(y_true[class_mask], y_pred[class_mask])
            )

            if has_both_classes:
                one_vs_rest_labels = (y_true == target_class).astype(np.int64)
                one_vs_rest_scores = (
                    probability if target_class == 1 else 1.0 - probability
                )
                per_class_auc.append(
                    roc_auc_score(one_vs_rest_labels, one_vs_rest_scores)
                )
            else:
                per_class_auc.append(float("nan"))

        accuracies[split_name] = float(np.mean(per_class_accuracy) * scale)
        finite_aucs = [value for value in per_class_auc if not np.isnan(value)]
        auc_rocs[split_name] = (
            float(np.mean(finite_aucs) * scale) if finite_aucs else float("nan")
        )

        sensitive_0 = split_sensitive == 0
        sensitive_1 = split_sensitive == 1
        demographic_parities[split_name] = (
            _absolute_rate_gap(y_pred, sensitive_0, sensitive_1) * scale
        )
        equal_opportunities[split_name] = (
            _absolute_rate_gap(
                y_pred,
                sensitive_0 & (y_true == 1),
                sensitive_1 & (y_true == 1),
            )
            * scale
        )

    return accuracies, auc_rocs, demographic_parities, equal_opportunities
