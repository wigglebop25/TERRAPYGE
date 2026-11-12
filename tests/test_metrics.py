"""Unit tests for threshold selection and thresholded metrics.

The decision threshold must be selected on validation data, so these tests cover
the selection helpers in isolation plus the metric reporting at a fixed
threshold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.terrapyge.utils.metrics import (
    f1_optimal_threshold,
    prevalence_matched_threshold,
    thresholded_metrics,
    youden_threshold,
)


def test_f1_optimal_threshold_separates_perfectly_separable_data():
    y = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.6, 0.4, 0.1])

    threshold = f1_optimal_threshold(y, probs)

    # any threshold in (0.4, 0.6] gives a perfect split
    assert 0.4 < threshold <= 0.6


def test_f1_optimal_threshold_beats_fixed_half_on_imbalanced_data():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.145).astype(int)
    # shift probabilities down so that 0.5 is a poor threshold
    probs = np.clip(0.2 + 0.4 * y + rng.normal(0, 0.1, y.size), 0.0, 1.0)

    threshold = f1_optimal_threshold(y, probs)
    calibrated = thresholded_metrics(y, probs, threshold)['f1']
    fixed = thresholded_metrics(y, probs, 0.5)['f1']

    assert calibrated > fixed


def test_f1_optimal_threshold_single_class_returns_default():
    assert f1_optimal_threshold(np.zeros(10, dtype=int), np.zeros(10)) == 0.5
    assert f1_optimal_threshold(np.ones(10, dtype=int), np.ones(10)) == 0.5


def test_youden_threshold_separates_perfectly_separable_data():
    y = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.6, 0.4, 0.1])

    threshold = youden_threshold(y, probs)

    assert 0.4 < threshold <= 0.6


def test_thresholded_metrics_hand_computed():
    y = np.array([1, 1, 0, 0])

    metrics = thresholded_metrics(y, np.array([0.9, 0.6, 0.4, 0.1]), 0.5)

    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(1.0)
    assert metrics['f1'] == pytest.approx(1.0)
    assert metrics['predicted_positive_rate'] == pytest.approx(0.5)


def test_thresholded_metrics_high_threshold_lowers_recall():
    y = np.array([1, 1, 0, 0])

    metrics = thresholded_metrics(y, np.array([0.9, 0.6, 0.4, 0.1]), 0.7)

    assert metrics['precision'] == pytest.approx(1.0)
    assert metrics['recall'] == pytest.approx(0.5)
    assert metrics['f1'] == pytest.approx(2 / 3, abs=1e-6)


def test_thresholded_metrics_reports_threshold():
    metrics = thresholded_metrics(np.array([1, 0]), np.array([0.9, 0.1]), 0.35)

    assert metrics['threshold'] == pytest.approx(0.35)


def test_prevalence_matched_threshold_flags_target_fraction():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.2).astype(int)
    probs = rng.random(2000)

    threshold = prevalence_matched_threshold(y, probs)
    flagged = float(np.mean(probs >= threshold))

    assert flagged == pytest.approx(0.2, abs=0.01)


def test_prevalence_matched_threshold_does_not_degenerate_on_weak_model():
    # a near-chance model: probabilities carry almost no signal
    rng = np.random.default_rng(2)
    y = (rng.random(1000) < 0.145).astype(int)
    probs = rng.random(1000)

    threshold = prevalence_matched_threshold(y, probs)
    metrics = thresholded_metrics(y, probs, threshold)

    # still predicts a sensible number of positives rather than none
    assert metrics['predicted_positive_rate'] > 0.05
    assert metrics['predicted_positive_rate'] < 0.30


def test_prevalence_matched_threshold_single_class_returns_default():
    assert prevalence_matched_threshold(np.zeros(10, dtype=int), np.zeros(10)) == 0.5


def test_prevalence_matched_threshold_all_positive_flags_everything():
    y = np.ones(10, dtype=int)
    probs = np.linspace(0.1, 0.9, 10)

    threshold = prevalence_matched_threshold(y, probs)

    assert np.all(probs >= threshold)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
