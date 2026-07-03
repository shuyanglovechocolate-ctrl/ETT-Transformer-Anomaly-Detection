"""Tests for anomaly residual aggregation (Module 6.1)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import aggregate_residuals


def _long_df():
    # target_date A predicted by 2 windows (h_idx 0 and 1); B by 1 window.
    return pd.DataFrame({
        "sample_index": [0, 1, 1],
        "horizon_index": [0, 1, 0],
        "target_date": ["2018-01-01 00:00", "2018-01-01 00:00", "2018-01-01 01:00"],
        "y_true": [10.0, 10.0, 20.0],
        "y_pred": [9.0, 6.0, 19.0],  # abs errors: 1, 4, 1
    })


def test_first_keeps_horizon_index_zero():
    out = aggregate_residuals(_long_df(), method="first")
    # Only h_idx==0 rows: target A (err 1) and target B (err 1).
    assert set(out["target_date"]) == {"2018-01-01 00:00", "2018-01-01 01:00"}
    assert out["target_date"].is_unique
    a = out[out.target_date == "2018-01-01 00:00"].iloc[0]
    assert a["anomaly_score"] == 1.0


def test_mean_aggregates_abs_residual():
    out = aggregate_residuals(_long_df(), method="mean")
    a = out[out.target_date == "2018-01-01 00:00"].iloc[0]
    # mean of abs errors 1 and 4 = 2.5
    assert a["anomaly_score"] == 2.5
    assert out["target_date"].is_unique


def test_max_aggregates_abs_residual():
    out = aggregate_residuals(_long_df(), method="max")
    a = out[out.target_date == "2018-01-01 00:00"].iloc[0]
    assert a["anomaly_score"] == 4.0  # max of 1 and 4


def test_output_columns_and_no_nan():
    out = aggregate_residuals(_long_df(), method="mean")
    expected = {"target_date", "y_true", "y_pred", "residual", "abs_residual",
                "anomaly_score", "aggregation_method"}
    assert expected.issubset(out.columns)
    assert not out[["anomaly_score", "abs_residual"]].isna().any().any()


def test_reusable_for_injected_residuals():
    # Simulate injection: modify y_true, same y_pred -> larger residual/score.
    clean = _long_df()
    injected = clean.copy()
    injected.loc[injected.target_date == "2018-01-01 01:00", "y_true"] = 40.0
    out = aggregate_residuals(injected, method="first")
    b = out[out.target_date == "2018-01-01 01:00"].iloc[0]
    assert b["anomaly_score"] == 21.0  # |40 - 19|


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        aggregate_residuals(_long_df(), method="median")


def test_missing_columns_raises():
    bad = pd.DataFrame({"target_date": ["x"], "y_true": [1.0]})
    with pytest.raises(ValueError):
        aggregate_residuals(bad, method="first")
