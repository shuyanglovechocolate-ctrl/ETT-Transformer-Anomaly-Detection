"""Forecasting evaluation in the original OT scale.

All metrics and residuals are computed AFTER inverse-transforming both
predictions and ground truth with ``scaler_y`` (the target scaler fitted only on
training OT). Computing MAE / RMSE / residuals in standardised units would make
them uninterpretable, so every function here enforces the original-scale rule.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def inverse_transform_predictions(y_scaled, scaler_y):
    """Inverse-transform scaled target values back to the original OT scale.

    Parameters
    ----------
    y_scaled : np.ndarray
        Scaled values with shape [N], [N, 1], or [N, horizon].
    scaler_y : sklearn scaler
        Target scaler fitted only on the training OT column.

    Returns
    -------
    np.ndarray
        Values in the original OT scale with the same shape as the input.
    """
    y_scaled = np.asarray(y_scaled)
    original_shape = y_scaled.shape

    y_reshaped = y_scaled.reshape(-1, 1)
    y_original = scaler_y.inverse_transform(y_reshaped)

    return y_original.reshape(original_shape)


def calculate_forecasting_metrics(y_true_scaled, y_pred_scaled, scaler_y):
    """Calculate MAE and RMSE in the original OT scale.

    Metrics are calculated after inverse-transforming both predictions and
    ground truth using scaler_y, ensuring interpretability and avoiding errors
    reported in standardised units.

    Parameters
    ----------
    y_true_scaled : np.ndarray
        Ground truth values in scaled space. Shape: [N, horizon]
    y_pred_scaled : np.ndarray
        Predicted values in scaled space. Shape: [N, horizon]
    scaler_y : sklearn scaler
        Target scaler fitted only on training OT.

    Returns
    -------
    dict
        MAE and RMSE calculated in the original OT scale.
    """
    y_true_original = inverse_transform_predictions(y_true_scaled, scaler_y)
    y_pred_original = inverse_transform_predictions(y_pred_scaled, scaler_y)

    y_true_flat = y_true_original.reshape(-1)
    y_pred_flat = y_pred_original.reshape(-1)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))

    return {
        "mae": mae,
        "rmse": rmse,
    }


def calculate_residuals(y_true_scaled, y_pred_scaled, scaler_y, absolute=True):
    """Calculate residuals in the original OT scale.

    Parameters
    ----------
    y_true_scaled : np.ndarray
        Ground truth values in scaled space. Shape: [N, horizon]
    y_pred_scaled : np.ndarray
        Predicted values in scaled space. Shape: [N, horizon]
    scaler_y : sklearn scaler
        Target scaler fitted only on training OT.
    absolute : bool
        If True, return absolute residuals. If False, return signed residuals.

    Returns
    -------
    np.ndarray
        Residuals in the original OT scale.
    """
    y_true_original = inverse_transform_predictions(y_true_scaled, scaler_y)
    y_pred_original = inverse_transform_predictions(y_pred_scaled, scaler_y)

    residuals = y_true_original - y_pred_original

    if absolute:
        residuals = np.abs(residuals)

    return residuals


def flatten_horizon_outputs(y):
    """Flatten [N, horizon] arrays into one-dimensional arrays.

    Useful for global MAE/RMSE calculation and residual distribution analysis
    across all forecasted time points.
    """
    return np.asarray(y).reshape(-1)
