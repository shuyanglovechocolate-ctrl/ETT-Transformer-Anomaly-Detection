"""Tests for the Module 3.1 minimal training pipeline.

Run with: pytest tests/
Uses small synthetic data and a fitted scaler so the tests stay fast.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import create_dataloader
from src.models import LinearForecaster, NaiveForecaster, count_parameters
from src.training import (
    train_one_epoch,
    evaluate_loss,
    predict,
    create_prediction_dataframe,
    save_checkpoint,
    load_checkpoint,
)

INPUT_LEN = 16
HORIZON = 4
NUM_FEATURES = 7
N = 64
DEVICE = torch.device("cpu")


def _loader():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, INPUT_LEN, NUM_FEATURES)).astype(np.float32)
    y = rng.normal(size=(N, HORIZON)).astype(np.float32)
    return create_dataloader(X, y, batch_size=16, shuffle=True)


def _scaler_y():
    s = StandardScaler()
    s.fit(np.random.default_rng(1).normal(size=(100, 1)))
    return s


def test_train_one_epoch_returns_float():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, _loader(), optimizer, nn.MSELoss(), DEVICE)
    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_training_loss_not_increasing_much():
    torch.manual_seed(0)
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _loader()
    criterion = nn.MSELoss()
    first = train_one_epoch(model, loader, optimizer, criterion, DEVICE)
    second = train_one_epoch(model, loader, optimizer, criterion, DEVICE)
    assert np.isfinite(first) and np.isfinite(second)
    # Stable convergence: second epoch should not blow up.
    assert second <= first + 1e-3


def test_evaluate_loss_returns_float():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    loss = evaluate_loss(model, _loader(), nn.MSELoss(), DEVICE)
    assert isinstance(loss, float) and np.isfinite(loss)


def test_predict_shape():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    y_true, y_pred = predict(model, _loader(), DEVICE)
    assert y_true.shape == (N, HORIZON)
    assert y_pred.shape == (N, HORIZON)


def test_create_prediction_dataframe_columns():
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    y_true, y_pred = predict(model, _loader(), DEVICE)
    dates = np.tile(
        np.arange(HORIZON), (N, 1)
    ).astype("datetime64[h]")  # dummy datetimes [N, horizon]
    df = create_prediction_dataframe(y_true, y_pred, dates, _scaler_y())
    expected = {
        "sample_index", "horizon_index", "target_date",
        "y_true", "y_pred", "residual", "abs_residual",
    }
    assert expected.issubset(df.columns)
    assert len(df) == N * HORIZON


def test_save_and_load_checkpoint(tmp_path):
    model = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, epoch=1, val_loss=0.5, path=str(path))
    assert path.exists()

    model2 = LinearForecaster(INPUT_LEN, NUM_FEATURES, HORIZON)
    ckpt = load_checkpoint(model2, str(path))
    assert ckpt["epoch"] == 1
    assert ckpt["val_loss"] == 0.5


def test_naive_has_no_parameters_so_no_optimizer():
    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    model = NaiveForecaster(INPUT_LEN, NUM_FEATURES, HORIZON, feature_cols)
    assert count_parameters(model) == 0
    # Building an Adam optimizer on an empty parameter list must fail, which is
    # exactly why the training entry point gates on count_parameters > 0.
    with pytest.raises(ValueError):
        torch.optim.Adam(model.parameters(), lr=1e-3)
