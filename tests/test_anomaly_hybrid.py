"""Tests for hybrid residual + flatness detection (Module 6.10)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    rank_normalize, hybrid_rankmax_score, detect_hybrid_or, score_detector,
    inject_synthetic_anomalies,
)


def test_rank_normalize_in_unit_range():
    ref = np.arange(0, 100, dtype=float)
    r = rank_normalize(np.array([-5.0, 50.0, 200.0]), ref)
    assert np.all((r >= 0) & (r <= 1))
    assert r[0] == 0.0          # below all reference
    assert r[2] == 1.0          # above all reference


def test_rank_normalize_depends_only_on_reference():
    scores = np.array([1.0, 2.0, 3.0])
    r_a = rank_normalize(scores, np.array([0.0, 2.0, 4.0]))
    r_b = rank_normalize(scores, np.array([0.0, 1.0, 10.0]))
    assert not np.allclose(r_a, r_b)  # same scores, different reference -> different ranks


def test_hybrid_rankmax_is_max_of_ranks():
    residual = np.array([1.0, 5.0])
    flatness = np.array([9.0, 1.0])
    val_r = np.arange(0, 10, dtype=float)
    val_f = np.arange(0, 10, dtype=float)
    h = hybrid_rankmax_score(residual, flatness, val_r, val_f)
    rr = rank_normalize(residual, val_r)
    rf = rank_normalize(flatness, val_f)
    assert np.allclose(h, np.maximum(rr, rf))


def test_detect_hybrid_or_union():
    residual = np.array([0.0, 5.0, 0.0])
    flatness = np.array([0.0, 0.0, 9.0])
    out = detect_hybrid_or(residual, flatness, residual_threshold=1.0,
                           flatness_threshold=1.0)
    assert out.tolist() == [False, True, True]  # union of the two alarms


def _val_and_injected(n=300):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="1h").astype(str)
    y = rng.normal(10.0, 1.0, size=n)
    yp = y + rng.normal(0.0, 0.2, size=n)
    val = pd.DataFrame({"target_date": dates, "y_true": y, "y_pred": yp,
                        "residual": y - yp, "abs_residual": np.abs(y - yp),
                        "anomaly_score": np.abs(y - yp)})
    injected = inject_synthetic_anomalies(val, "frozen", anomaly_ratio=0.05,
                                          duration_range=(12, 24), seed=42)
    return val, injected


def test_flatness_detector_scores_finite_and_nonneg():
    val, injected = _val_and_injected()
    vs, ts = score_detector("flatness", injected, val)
    assert len(ts) == len(injected)
    assert np.all(np.isfinite(ts)) and np.all(ts >= 0)


def test_flatness_higher_on_frozen_points():
    val, injected = _val_and_injected()
    _, ts = score_detector("flatness", injected, val)
    labels = injected["is_anomaly"].to_numpy()
    # Frozen points should on average have much higher flatness score.
    assert ts[labels].mean() > ts[~labels].mean()


def test_hybrid_rankmax_scores_in_unit_range():
    val, injected = _val_and_injected()
    _, ts = score_detector("hybrid_rankmax", injected, val)
    assert np.all((ts >= 0) & (ts <= 1))


def test_rank_normalize_empty_reference_raises():
    with pytest.raises(ValueError):
        rank_normalize(np.array([1.0]), np.array([]))
