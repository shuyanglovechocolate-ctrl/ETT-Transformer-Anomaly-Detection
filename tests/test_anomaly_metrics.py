"""Tests for point-wise anomaly detection metrics (Module 6.4)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import calculate_detection_metrics


def _df(labels, preds):
    return pd.DataFrame({"is_anomaly": labels, "predicted_anomaly": preds})


def test_perfect_detection():
    m = calculate_detection_metrics(_df([True, True, False, False],
                                        [True, True, False, False]))
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["false_positive_rate"] == 0.0
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_all_missed():
    m = calculate_detection_metrics(_df([True, True, False], [False, False, False]))
    assert m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["fn"] == 2


def test_false_positives_lower_precision():
    m = calculate_detection_metrics(_df([True, False, False, False],
                                        [True, True, True, False]))
    # TP=1, FP=2 -> precision 1/3; recall 1/1 = 1
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == 1.0
    assert m["false_positive_rate"] > 0


def test_no_predicted_anomalies():
    m = calculate_detection_metrics(_df([True, False], [False, False]))
    assert m["precision"] == 0.0 and m["recall"] == 0.0
    assert m["num_predicted_anomaly"] == 0


def test_no_true_anomalies():
    # No positives: recall 0 (no TP possible), fpr computed normally.
    m = calculate_detection_metrics(_df([False, False, False], [True, False, False]))
    assert m["recall"] == 0.0
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["num_true_anomaly"] == 0


def test_counts_sum_to_num_points():
    m = calculate_detection_metrics(_df([True, False, True, False],
                                        [True, True, False, False]))
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == m["num_points"] == 4


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        calculate_detection_metrics(pd.DataFrame({"predicted_anomaly": [True]}))
    with pytest.raises(ValueError):
        calculate_detection_metrics(pd.DataFrame({"is_anomaly": [True]}))


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        calculate_detection_metrics(_df([], []))


def test_nan_raises():
    df = pd.DataFrame({"is_anomaly": [True, np.nan], "predicted_anomaly": [True, False]})
    with pytest.raises(ValueError):
        calculate_detection_metrics(df)
