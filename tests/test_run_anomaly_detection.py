"""Tests for the end-to-end anomaly detection experiment (Module 6.5)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.run_anomaly_detection import evaluate_scenario, RESULT_COLUMNS


def _clean_series(n=300):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="1h").astype(str)
    y_true = rng.normal(10.0, 1.0, size=n)
    # small clean residuals -> low baseline anomaly score
    y_pred = y_true + rng.normal(0.0, 0.2, size=n)
    return pd.DataFrame({"target_date": dates, "y_true": y_true, "y_pred": y_pred})


def test_evaluate_scenario_spike_high_recall():
    test = _clean_series()
    val_scores = np.abs(np.random.default_rng(1).normal(0.0, 0.2, size=300))
    detected, m = evaluate_scenario(val_scores, test, "spike", "percentile")
    # spike magnitude 3*std swamps the small clean residuals -> all detected
    assert m["recall"] == 1.0
    assert m["threshold_value"] > 0
    assert "predicted_anomaly" in detected.columns


def test_evaluate_scenario_returns_all_metric_keys():
    test = _clean_series()
    val_scores = np.abs(np.random.default_rng(2).normal(0.0, 0.2, size=300))
    _, m = evaluate_scenario(val_scores, test, "level_shift", "mad")
    for key in ("precision", "recall", "f1", "false_positive_rate",
                "true_negative_rate", "tp", "fp", "tn", "fn", "threshold_value"):
        assert key in m


def test_frozen_harder_than_spike():
    test = _clean_series()
    val_scores = np.abs(np.random.default_rng(3).normal(0.0, 0.2, size=300))
    _, spike = evaluate_scenario(val_scores, test, "spike", "percentile")
    _, frozen = evaluate_scenario(val_scores, test, "frozen", "percentile")
    # Frozen anomalies produce weaker residual signals than spikes.
    assert frozen["recall"] <= spike["recall"]


def test_result_columns_defined():
    # The results schema must include the key reporting fields.
    for col in ("dataset", "model", "horizon", "aggregation_method",
                "anomaly_type", "threshold_method", "precision", "recall", "f1"):
        assert col in RESULT_COLUMNS
