"""Prediction collection and prediction-dataframe export.

The exported dataframe keeps the full windowed output (one row per
sample x horizon step), including ``sample_index``, ``horizon_index`` and
``target_date``. Because the sliding window has stride 1, a single timestamp is
predicted by multiple windows; preserving these columns lets Module 4 choose how
to reduce the overlap (e.g. one-step-ahead only, or aggregate per date).
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluate import inverse_transform_predictions


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader, returning scaled (y_true, y_pred).

    Returns
    -------
    y_true_scaled, y_pred_scaled : np.ndarray
        Both shape [num_samples, horizon] in scaled space.
    """
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y_pred = model(x)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(y_pred.cpu().numpy())

    y_true_scaled = np.concatenate(y_true_list, axis=0)
    y_pred_scaled = np.concatenate(y_pred_list, axis=0)
    return y_true_scaled, y_pred_scaled


def create_prediction_dataframe(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_dates: np.ndarray,
    scaler_y,
) -> pd.DataFrame:
    """Build a long-format prediction dataframe in the original OT scale.

    Columns: sample_index, horizon_index, target_date, y_true, y_pred,
    residual, abs_residual.
    """
    num_samples, horizon = y_true_scaled.shape

    y_true = inverse_transform_predictions(y_true_scaled, scaler_y)
    y_pred = inverse_transform_predictions(y_pred_scaled, scaler_y)

    sample_index = np.repeat(np.arange(num_samples), horizon)
    horizon_index = np.tile(np.arange(horizon), num_samples)
    target_date = np.asarray(y_dates).reshape(-1)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    residual = y_true_flat - y_pred_flat

    return pd.DataFrame(
        {
            "sample_index": sample_index,
            "horizon_index": horizon_index,
            "target_date": target_date,
            "y_true": y_true_flat,
            "y_pred": y_pred_flat,
            "residual": residual,
            "abs_residual": np.abs(residual),
        }
    )
