"""Tests for threshold-based anomaly detection (Module 6.3)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import detect_anomalies


def _df():
    return pd.DataFrame({
        "target_date": ["a", "b", "c", "d"],
        "y_true_anomalous": [10.0, 20.0, 10.0, 30.0],
        "y_pred": [10.0, 10.0, 10.0, 10.0],
        "anomaly_score": [0.5, 5.0, 2.0, 8.0],
    })


def test_detect_above_threshold_flagged():
    out = detect_anomalies(_df(), threshold=2.0)
    # strict '>' : score 2.0 is NOT flagged, 5.0 and 8.0 are
    assert out["predicted_anomaly"].tolist() == [False, True, False, True]


def test_threshold_written_to_every_row():
    out = detect_anomalies(_df(), threshold=3.5)
    assert (out["threshold"] == 3.5).all()


def test_predicted_anomaly_is_bool():
    out = detect_anomalies(_df(), threshold=1.0)
    assert out["predicted_anomaly"].dtype == bool


def test_input_not_modified():
    src = _df()
    _ = detect_anomalies(src, threshold=1.0)
    assert "predicted_anomaly" not in src.columns
    assert "threshold" not in src.columns


def test_missing_score_column_raises():
    df = _df().drop(columns=["anomaly_score"])
    with pytest.raises(ValueError):
        detect_anomalies(df, threshold=1.0)


def test_negative_threshold_raises():
    with pytest.raises(ValueError):
        detect_anomalies(_df(), threshold=-1.0)


def test_nan_score_raises():
    df = _df()
    df.loc[0, "anomaly_score"] = np.nan
    with pytest.raises(ValueError):
        detect_anomalies(df, threshold=1.0)
