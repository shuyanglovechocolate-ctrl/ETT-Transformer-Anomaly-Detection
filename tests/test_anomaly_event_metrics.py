"""Tests for event-wise anomaly detection metrics (Module 6.7)."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import calculate_event_detection_metrics


def _df(seg, pred):
    """Build a df; is_anomaly = (seg > 0)."""
    seg = np.asarray(seg)
    return pd.DataFrame({
        "is_anomaly": seg > 0,
        "predicted_anomaly": np.asarray(pred, dtype=bool),
        "anomaly_segment_id": seg,
    })


def test_single_event_detected_delay_zero():
    # segment 1 spans idx 2..4; detected at its first point -> delay 0
    m = calculate_event_detection_metrics(
        _df([0, 0, 1, 1, 1, 0], [False, False, True, False, False, False]))
    assert m["num_true_events"] == 1
    assert m["num_detected_events"] == 1
    assert m["event_recall"] == 1.0
    assert m["mean_detection_delay"] == 0.0


def test_detection_delay_in_middle():
    # segment 1 spans idx 1..4; first detection at idx 3 -> delay 2
    m = calculate_event_detection_metrics(
        _df([0, 1, 1, 1, 1], [False, False, False, True, False]))
    assert m["mean_detection_delay"] == 2.0
    assert m["max_detection_delay"] == 2


def test_partial_event_recall():
    # two events; only the first is detected
    seg = [1, 1, 0, 2, 2]
    pred = [True, False, False, False, False]
    m = calculate_event_detection_metrics(_df(seg, pred))
    assert m["num_true_events"] == 2
    assert m["num_detected_events"] == 1
    assert m["event_recall"] == 0.5


def test_fully_missed_event_delay_nan():
    m = calculate_event_detection_metrics(
        _df([0, 1, 1, 0], [False, False, False, False]))
    assert m["event_recall"] == 0.0
    assert math.isnan(m["mean_detection_delay"])


def test_false_positive_in_normal_does_not_affect_event_recall():
    # anomaly event detected; extra FP in a normal region is irrelevant here
    m = calculate_event_detection_metrics(
        _df([0, 1, 1, 0, 0], [True, True, False, True, False]))
    assert m["event_recall"] == 1.0


def test_no_true_events():
    m = calculate_event_detection_metrics(
        _df([0, 0, 0], [True, False, False]))
    assert m["num_true_events"] == 0
    assert m["event_recall"] == 0.0
    assert math.isnan(m["mean_detection_delay"])


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        calculate_event_detection_metrics(pd.DataFrame({"is_anomaly": [True]}))


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        calculate_event_detection_metrics(_df([], []))


def test_nan_prediction_raises():
    df = _df([1, 1], [True, False])
    df.loc[0, "predicted_anomaly"] = np.nan
    with pytest.raises(ValueError):
        calculate_event_detection_metrics(df)
