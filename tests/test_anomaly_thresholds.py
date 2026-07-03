"""Tests for anomaly threshold computation (Module 6.3)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import compute_threshold


def test_percentile_threshold():
    scores = np.arange(0, 101, dtype=float)  # 0..100
    assert compute_threshold(scores, "percentile", percentile=99) == pytest.approx(99.0)


def test_mean_std_threshold():
    scores = np.array([0.0, 2.0, 4.0])  # mean 2, std 1.632993...
    expected = scores.mean() + 3.0 * scores.std(ddof=0)
    assert compute_threshold(scores, "mean_std", k=3.0) == pytest.approx(expected)


def test_iqr_threshold():
    scores = np.arange(0, 101, dtype=float)
    q1, q3 = np.percentile(scores, [25, 75])
    assert compute_threshold(scores, "iqr", k=1.5) == pytest.approx(q3 + 1.5 * (q3 - q1))


def test_mad_threshold():
    scores = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    assert compute_threshold(scores, "mad", k=3.5) == pytest.approx(median + 3.5 * 1.4826 * mad)


def test_thresholds_ordering_reasonable():
    # A robust MAD/percentile threshold sits above the bulk of the scores.
    scores = np.abs(np.random.default_rng(0).normal(size=1000))
    for m in ("percentile", "mean_std", "iqr", "mad"):
        thr = compute_threshold(scores, m)
        assert thr > np.median(scores)


def test_empty_scores_raises():
    with pytest.raises(ValueError):
        compute_threshold(np.array([]), "percentile")


def test_nan_scores_raises():
    with pytest.raises(ValueError):
        compute_threshold(np.array([1.0, np.nan, 2.0]), "mean_std")


def test_negative_scores_raises():
    with pytest.raises(ValueError):
        compute_threshold(np.array([1.0, -2.0]), "iqr")


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        compute_threshold(np.array([1.0, 2.0]), "zscore")


def test_invalid_percentile_and_k_raise():
    with pytest.raises(ValueError):
        compute_threshold(np.array([1.0, 2.0]), "percentile", percentile=150)
    with pytest.raises(ValueError):
        compute_threshold(np.array([1.0, 2.0]), "mean_std", k=-1)
