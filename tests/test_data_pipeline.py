"""Minimal tests for the Module 1 data pipeline.

Run with: pytest tests/
These use a small synthetic ETT-like dataframe so they do not depend on the
real CSV files and stay fast.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.splitter import time_based_split
from src.data.preprocessing import prepare_scaled_splits
from src.data.dataset import create_sliding_windows, prepare_windowed_dataloaders
from src.utils.config import validate_config


ETT_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def make_synthetic_df(n: int = 2000) -> pd.DataFrame:
    """Build a small hourly ETT-like dataframe."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="1h")
    data = {"date": dates}
    for col in ETT_FEATURES:
        data[col] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_split_ratio():
    n = 1000
    df = make_synthetic_df(n)
    train_df, val_df, test_df = time_based_split(df, 0.7, 0.1, 0.2)
    # No rows are lost or duplicated across splits.
    assert len(train_df) + len(val_df) + len(test_df) == n
    # Lengths match the configured ratios (allow off-by-one float rounding).
    assert abs(len(train_df) - 700) <= 1
    assert abs(len(val_df) - 100) <= 1
    assert abs(len(test_df) - 200) <= 1
    # Chronological order is preserved (no overlap, train before val before test).
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()


def test_scaler_y_single_feature():
    df = make_synthetic_df()
    train_df, val_df, test_df = time_based_split(df)
    scaled = prepare_scaled_splits(train_df, val_df, test_df, "multivariate")
    assert scaled["scaler_y"].n_features_in_ == 1


def test_univariate_num_features():
    df = make_synthetic_df()
    train_df, val_df, test_df = time_based_split(df)
    scaled = prepare_scaled_splits(train_df, val_df, test_df, "univariate")
    assert scaled["num_features"] == 1


def test_multivariate_num_features():
    df = make_synthetic_df()
    train_df, val_df, test_df = time_based_split(df)
    scaled = prepare_scaled_splits(train_df, val_df, test_df, "multivariate")
    assert scaled["num_features"] == 7
    assert scaled["scaler_x"].n_features_in_ == 7


def test_sliding_window_shape_and_dates():
    df = make_synthetic_df(500)
    dates = df["date"].to_numpy()
    data_x = df[ETT_FEATURES].to_numpy()
    data_y = df[["OT"]].to_numpy()

    X, y, y_dates = create_sliding_windows(
        data_x, data_y, dates, input_len=96, horizon=24
    )

    expected_samples = 500 - 96 - 24 + 1
    assert X.shape == (expected_samples, 96, 7)
    assert y.shape == (expected_samples, 24)
    assert y_dates.shape == (expected_samples, 24)
    # First target date is one step after the first input window ends.
    assert y_dates[0, 0] == dates[96]


def test_y_dates_alignment():
    """Each window's target dates must line up exactly with the source series.

    Downstream residual plots, dashboards and anomaly markers rely on this, so
    the alignment is asserted explicitly rather than only checked by hand.
    """
    n, input_len, horizon = 500, 96, 24
    df = make_synthetic_df(n)
    dates = df["date"].to_numpy()
    data_x = df[ETT_FEATURES].to_numpy()
    data_y = df[["OT"]].to_numpy()

    _, _, y_dates = create_sliding_windows(
        data_x, data_y, dates, input_len=input_len, horizon=horizon
    )

    # First sample: targets are t+1 .. t+horizon right after the input window.
    assert y_dates[0, 0] == dates[input_len]
    assert y_dates[0, -1] == dates[input_len + horizon - 1]

    # An arbitrary interior sample must hold the same relationship.
    i = 100
    assert y_dates[i, 0] == dates[i + input_len]
    assert y_dates[i, -1] == dates[i + input_len + horizon - 1]

    # Within a window the target dates are strictly increasing and contiguous.
    step = dates[1] - dates[0]
    assert np.all(np.diff(y_dates[0]) == step)


def test_boundary_handling_no_leakage():
    # Large enough that every split (incl. the 10% val) can form windows.
    df = make_synthetic_df(3000)
    train_df, val_df, test_df = time_based_split(df)
    scaled = prepare_scaled_splits(train_df, val_df, test_df, "multivariate")
    windowed = prepare_windowed_dataloaders(scaled, input_len=96, horizon=24)

    assert windowed["train_X"].shape[0] == len(train_df) - 96 - 24 + 1
    assert windowed["val_X"].shape[0] == len(val_df) - 96 - 24 + 1
    assert windowed["test_X"].shape[0] == len(test_df) - 96 - 24 + 1


def test_validate_config_rejects_bad_input_type():
    config = {
        "dataset": {"input_type": "bogus", "path": "x", "target": "OT", "name": "X"},
        "split": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2},
        "window": {"input_len": 96, "horizon": 24},
        "training": {"batch_size": 64, "seed": 42},
    }
    with pytest.raises(ValueError):
        validate_config(config)


def test_validate_config_rejects_bad_ratios():
    config = {
        "dataset": {"input_type": "multivariate", "path": "x", "target": "OT", "name": "X"},
        "split": {"train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.2},
        "window": {"input_len": 96, "horizon": 24},
        "training": {"batch_size": 64, "seed": 42},
    }
    with pytest.raises(ValueError):
        validate_config(config)
