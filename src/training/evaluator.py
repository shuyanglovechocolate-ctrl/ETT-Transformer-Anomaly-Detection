"""Loss evaluation and original-scale forecasting metrics."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluate import calculate_forecasting_metrics, inverse_transform_predictions


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Return the average (scaled-space) loss over a data loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def compute_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler_y,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute MAE, RMSE and WAPE in the original OT scale.

    MAE / RMSE reuse src.evaluate.calculate_forecasting_metrics. We report WAPE
    (Weighted Absolute Percentage Error = sum|err| / sum|y_true|) instead of MAPE
    because ETT oil temperature crosses zero, which makes per-point MAPE explode
    even with an epsilon. WAPE aggregates before dividing and stays stable.
    """
    metrics = calculate_forecasting_metrics(y_true_scaled, y_pred_scaled, scaler_y)

    y_true = inverse_transform_predictions(y_true_scaled, scaler_y).reshape(-1)
    y_pred = inverse_transform_predictions(y_pred_scaled, scaler_y).reshape(-1)
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps) * 100.0

    metrics["wape"] = float(wape)
    return metrics
