"""Threshold selection and thresholded classification metrics.

Landslide susceptibility labels are imbalanced (14.55% positive), so a fixed 0.5
probability threshold predicts almost no positives and reports F1 = 0 even for
models with high AUC. These helpers select a decision threshold on validation
data only, then report precision, recall and F1 at that threshold on held-out
data. Selecting the threshold on the test split would make the figures
optimistic in a new way, so callers must pass validation data.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def f1_optimal_threshold(y_true, probs, default=0.5):
    """Return the probability threshold that maximises F1 on the given data.

    Args:
        y_true: binary labels.
        probs: positive-class probabilities.
        default: returned when the data has a single class or no thresholds.

    Returns:
        The threshold as a float. Predictions are ``probs >= threshold``.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float(default)

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if thresholds.size == 0:
        return float(default)

    # precision/recall have one more element than thresholds; the final point is
    # the trivial all-positive prediction and has no corresponding threshold.
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[min(best, thresholds.size - 1)])


def youden_threshold(y_true, probs, default=0.5):
    """Return the threshold maximising Youden's J (sensitivity + specificity - 1)."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float(default)

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    if thresholds.size == 0:
        return float(default)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def prevalence_matched_threshold(y_true, probs, default=0.5):
    """Return the threshold that flags the same positive fraction as ``y_true``.

    This is the default rule used by the experiments. Unlike the F1-optimal
    rule, it stays well behaved for models with little signal: a near-chance
    model still flags roughly the observed prevalence, so precision and recall
    degrade gracefully instead of collapsing to zero predictions at an extreme
    threshold. It also matches the operational framing of the hazard map, where
    the intent is to prioritise a fixed proportion of slope units.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    if y_true.size == 0 or probs.size == 0:
        return float(default)

    target = float(np.mean(y_true))
    if target <= 0.0:
        return float(default)
    if target >= 1.0:
        return float(np.min(probs))

    return float(np.quantile(probs, 1.0 - target))


def thresholded_metrics(y_true, probs, threshold):
    """Precision, recall, F1 and predicted positive rate at ``threshold``."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    predicted = (probs >= threshold).astype(int)

    return {
        'threshold': float(threshold),
        'precision': float(precision_score(y_true, predicted, zero_division=0)),
        'recall': float(recall_score(y_true, predicted, zero_division=0)),
        'f1': float(f1_score(y_true, predicted, zero_division=0)),
        'predicted_positive_rate': float(predicted.mean()) if predicted.size else 0.0,
    }


__all__ = [
    'f1_optimal_threshold',
    'prevalence_matched_threshold',
    'youden_threshold',
    'thresholded_metrics',
]
