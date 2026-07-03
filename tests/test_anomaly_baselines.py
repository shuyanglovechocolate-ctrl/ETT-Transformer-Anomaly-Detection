"""Tests for causal statistical anomaly baselines (Module 6.6)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import compute_baseline_scores


def test_diff_score_flags_spike():
    y = np.full(50, 10.0)
    y[25] = 30.0  # spike
    scores = compute_baseline_scores("diff_score", y)
    assert scores.argmax() == 25  # largest jump at the spike
    assert np.all(scores >= 0)


def test_raw_zscore_uses_reference_stats():
    ref = np.array([0.0, 10.0])  # mean 5, std 5
    y = np.array([5.0, 15.0])
    scores = compute_baseline_scores("raw_zscore", y, ref_series=ref)
    # |5-5|/5 = 0 ; |15-5|/5 = 2
    assert scores[0] == pytest.approx(0.0, abs=1e-6)
    assert scores[1] == pytest.approx(2.0, abs=1e-4)


def test_rolling_zscore_is_causal():
    # Changing a value at index k must not affect scores before k.
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    base = compute_baseline_scores("rolling_zscore", y, window=10)
    y2 = y.copy()
    y2[60] += 50.0  # perturb the future
    perturbed = compute_baseline_scores("rolling_zscore", y2, window=10)
    assert np.allclose(base[:60], perturbed[:60])


def test_rolling_zscore_finite_on_frozen_segment():
    # A frozen (constant) segment gives past_std=0; eps must keep scores finite.
    y = np.concatenate([np.full(30, 5.0), np.full(30, 5.0)])
    scores = compute_baseline_scores("rolling_zscore", y, window=10)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


def test_scores_non_negative():
    y = np.random.default_rng(1).normal(size=200)
    for det in ("diff_score", "rolling_zscore"):
        s = compute_baseline_scores(det, y)
        assert np.all(s >= 0)


def test_raw_zscore_requires_ref():
    with pytest.raises(ValueError):
        compute_baseline_scores("raw_zscore", np.array([1.0, 2.0]))


def test_invalid_detector_raises():
    with pytest.raises(ValueError):
        compute_baseline_scores("bogus", np.array([1.0, 2.0]))
