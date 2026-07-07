"""Tests for threshold-free metrics and shared detector scoring (Module 6.9)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    pr_auc, roc_auc, best_f1_threshold, f1_at_target_fpr,
    calculate_threshold_free_metrics, score_detector, DETECTOR_TYPES,
    inject_synthetic_anomalies,
)


# ---------------------------------------------------------------------------
# Threshold-free metrics
# ---------------------------------------------------------------------------

def test_perfect_separation_aucs_are_one():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8])  # anomalies clearly higher
    assert pr_auc(labels, scores) == pytest.approx(1.0)
    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert best_f1_threshold(labels, scores)["best_f1"] == pytest.approx(1.0)


def test_roc_auc_random_is_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=2000)
    scores = rng.normal(size=2000)  # unrelated to labels
    assert roc_auc(labels, scores) == pytest.approx(0.5, abs=0.05)


def test_pr_auc_undefined_without_both_classes():
    assert np.isnan(pr_auc(np.zeros(5), np.arange(5.0)))
    assert np.isnan(pr_auc(np.ones(5), np.arange(5.0)))


def test_f1_at_target_fpr_respects_constraint():
    # 90 normals, 10 anomalies with clearly higher scores
    rng = np.random.default_rng(1)
    labels = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([rng.normal(0, 1, 90), rng.normal(6, 0.5, 10)])
    res = f1_at_target_fpr(labels, scores, target_fpr=0.02)
    assert res["fpr"] <= 0.02 + 1e-9
    assert res["recall"] == pytest.approx(1.0)  # well-separated -> all anomalies caught


def test_calculate_threshold_free_metrics_keys():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    m = calculate_threshold_free_metrics(labels, scores)
    for key in ("pr_auc", "roc_auc", "best_f1", "f1_at_fpr_1pct",
                "recall_at_fpr_5pct"):
        assert key in m


def test_validation_errors():
    with pytest.raises(ValueError):
        pr_auc(np.array([0, 1]), np.array([0.1]))  # length mismatch
    with pytest.raises(ValueError):
        pr_auc(np.array([]), np.array([]))  # empty
    with pytest.raises(ValueError):
        roc_auc(np.array([0, 1]), np.array([0.1, np.nan]))  # NaN score


# ---------------------------------------------------------------------------
# Shared detector scoring
# ---------------------------------------------------------------------------

def _val_and_injected(n=300):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="1h").astype(str)
    y = rng.normal(10.0, 1.0, size=n)
    yp = y + rng.normal(0.0, 0.2, size=n)
    val = pd.DataFrame({"target_date": dates, "y_true": y, "y_pred": yp,
                        "residual": y - yp, "abs_residual": np.abs(y - yp),
                        "anomaly_score": np.abs(y - yp)})
    injected = inject_synthetic_anomalies(val, "spike", anomaly_ratio=0.03,
                                          duration_range=(1, 3), seed=42)
    return val, injected


def test_score_detector_all_types():
    val, injected = _val_and_injected()
    for detector in DETECTOR_TYPES:
        vs, ts = score_detector(detector, injected, val)
        assert len(ts) == len(injected)
        assert np.all(np.isfinite(ts)) and np.all(ts >= 0)


def test_score_detector_residual_uses_anomaly_score():
    val, injected = _val_and_injected()
    _, ts = score_detector("residual", injected, val)
    assert np.allclose(ts, injected["anomaly_score"].to_numpy())


def test_score_detector_unknown_raises():
    val, injected = _val_and_injected()
    with pytest.raises(ValueError):
        score_detector("bogus", injected, val)
