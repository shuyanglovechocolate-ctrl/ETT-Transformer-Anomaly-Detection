"""Tests for the end-to-end anomaly detection experiment (Modules 6.5 / 6.6)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import inject_synthetic_anomalies
from experiments.run_anomaly_detection import evaluate, detector_scores, RESULT_COLUMNS, DETECTORS


def _residual_series(n=300):
    """A clean aggregated-residual series (as produced by Module 6.1)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="1h").astype(str)
    y_true = rng.normal(10.0, 1.0, size=n)
    y_pred = y_true + rng.normal(0.0, 0.2, size=n)  # small clean residuals
    return pd.DataFrame({
        "target_date": dates, "y_true": y_true, "y_pred": y_pred,
        "residual": y_true - y_pred, "abs_residual": np.abs(y_true - y_pred),
        "anomaly_score": np.abs(y_true - y_pred),
    })


def _inject(series, atype="spike", seed=42):
    dr = (1, 3) if atype == "spike" else (12, 24)
    return inject_synthetic_anomalies(series, atype, anomaly_ratio=0.03,
                                      duration_range=dr, seed=seed)


def test_all_detectors_run_and_return_metrics():
    val = _residual_series()
    injected = _inject(val, "spike")
    for detector in DETECTORS:
        _, m = evaluate(detector, injected, val, "percentile")
        for key in ("precision", "recall", "f1", "threshold_value"):
            assert key in m


def test_detectors_share_same_injected_set():
    # Fairness: all detectors evaluated on the SAME injected labels.
    val = _residual_series()
    injected = _inject(val, "spike")
    labels = injected["is_anomaly"].to_numpy()
    for detector in DETECTORS:
        detected, _ = evaluate(detector, injected, val, "percentile")
        assert np.array_equal(detected["is_anomaly"].to_numpy(), labels)


def test_residual_detects_spike():
    val = _residual_series()
    injected = _inject(val, "spike")
    _, m = evaluate("residual", injected, val, "percentile")
    assert m["recall"] == 1.0


def test_residual_beats_diff_baseline_on_spike():
    # The residual detector has a much tighter normal baseline (small clean
    # residuals) than the model-free diff baseline (natural OT volatility), so it
    # detects 3-sigma spikes more reliably. This is the key research comparison.
    val = _residual_series()
    injected = _inject(val, "spike")
    _, res = evaluate("residual", injected, val, "percentile")
    _, diff = evaluate("diff_score", injected, val, "percentile")
    assert res["recall"] >= diff["recall"]


def test_detector_scores_shapes_match():
    val = _residual_series()
    injected = _inject(val, "level_shift")
    for detector in DETECTORS:
        vs, ts = detector_scores(detector, injected, val)
        assert len(ts) == len(injected)
        assert np.all(np.isfinite(ts)) and np.all(ts >= 0)


def test_result_columns_include_detector_and_seed():
    for col in ("detector_type", "injection_seed", "precision", "recall", "f1"):
        assert col in RESULT_COLUMNS
