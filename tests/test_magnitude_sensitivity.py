"""Tests for the magnitude sensitivity control experiment (Module 6.7)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import inject_synthetic_anomalies
from experiments.run_anomaly_detection import evaluate
from experiments.run_magnitude_sensitivity import (
    MAGNITUDE_SCALES, SWEEP_ANOMALY_TYPES, RESULT_COLUMNS,
)


def _residual_series(n=400):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="1h").astype(str)
    y_true = rng.normal(10.0, 1.0, size=n)
    y_pred = y_true + rng.normal(0.0, 0.2, size=n)
    return pd.DataFrame({
        "target_date": dates, "y_true": y_true, "y_pred": y_pred,
        "residual": y_true - y_pred, "abs_residual": np.abs(y_true - y_pred),
        "anomaly_score": np.abs(y_true - y_pred),
    })


def test_config_sane():
    assert 1.0 in MAGNITUDE_SCALES and 5.0 in MAGNITUDE_SCALES
    assert "spike" in SWEEP_ANOMALY_TYPES and "level_shift" in SWEEP_ANOMALY_TYPES
    assert "frozen" not in SWEEP_ANOMALY_TYPES  # magnitude-independent
    for col in ("magnitude_scale", "detector_type", "f1"):
        assert col in RESULT_COLUMNS


def test_residual_recall_increases_with_magnitude():
    # Larger anomalies must be at least as detectable by the residual detector.
    val = _residual_series()
    recalls = []
    for mag in (1.0, 3.0, 5.0):
        injected = inject_synthetic_anomalies(
            val, "spike", anomaly_ratio=0.03, duration_range=(1, 3),
            magnitude_scale=mag, seed=42)
        _, m = evaluate("residual", injected, val, "percentile")
        recalls.append(m["recall"])
    assert recalls[0] <= recalls[1] <= recalls[2]
