"""Tests for the extended anomaly-type stress test (Module 7.7)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import inject_synthetic_anomalies
from experiments.run_extended_anomaly_types import (
    evaluate_detector,
    EXTENDED_ANOMALY_TYPES,
    EXTENDED_DETECTORS,
    EXTENDED_DURATION_RANGES,
)

METRIC_KEYS = {"average_precision", "roc_auc", "oracle_best_f1",
               "event_recall", "mean_detection_delay"}


def _val_and_injected(anomaly_type="drift", n=300, seed=42):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="1h").astype(str)
    y = rng.normal(10.0, 1.0, size=n)
    yp = y + rng.normal(0.0, 0.2, size=n)
    val = pd.DataFrame({"target_date": dates, "y_true": y, "y_pred": yp,
                        "residual": y - yp, "abs_residual": np.abs(y - yp),
                        "anomaly_score": np.abs(y - yp)})
    injected = inject_synthetic_anomalies(
        val, anomaly_type, anomaly_ratio=0.05,
        duration_range=EXTENDED_DURATION_RANGES[anomaly_type], seed=seed)
    return val, injected


def test_extended_config_is_sane():
    assert EXTENDED_ANOMALY_TYPES == ["drift", "noise_burst", "stuck_with_jitter"]
    assert "residual" in EXTENDED_DETECTORS and "flatness" in EXTENDED_DETECTORS
    for t in EXTENDED_ANOMALY_TYPES:
        lo, hi = EXTENDED_DURATION_RANGES[t]
        assert 1 <= lo <= hi


def test_evaluate_detector_keys_and_ranges():
    val, injected = _val_and_injected("drift")
    m = evaluate_detector(injected, val, "residual")
    assert set(m) == METRIC_KEYS
    assert 0.0 <= m["average_precision"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["event_recall"] <= 1.0
    assert np.isnan(m["mean_detection_delay"]) or m["mean_detection_delay"] >= 0


def test_all_detectors_run_on_every_extended_type():
    for anomaly_type in EXTENDED_ANOMALY_TYPES:
        val, injected = _val_and_injected(anomaly_type)
        for detector in EXTENDED_DETECTORS:
            m = evaluate_detector(injected, val, detector)
            # both classes are present, so PR-AUC is defined
            assert not np.isnan(m["average_precision"])
            assert 0.0 <= m["event_recall"] <= 1.0


def test_residual_detects_large_drift():
    # A drift ramp of magnitude 3*std should be recoverable by the residual score.
    val, injected = _val_and_injected("drift", seed=7)
    m = evaluate_detector(injected, val, "residual")
    assert m["average_precision"] > 0.3  # clearly better than random on ~5% base rate
