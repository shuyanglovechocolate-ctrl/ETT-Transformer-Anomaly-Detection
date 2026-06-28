"""Module 1 interface check.

Proves that Module 1 produces the downstream-ready outputs
(train_loader / val_loader / test_loader / scaler_y / num_features) and that the
evaluation layer can consume them to compute original-scale MAE / RMSE /
residuals. The "prediction" here is a dummy (zeros) used only to validate the
interface, not a real model.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_ett_dataset
from src.data.splitter import time_based_split, print_split_summary
from src.data.preprocessing import prepare_scaled_splits, print_preprocessing_summary
from src.data.dataset import prepare_windowed_dataloaders, print_window_summary
from src.evaluate import calculate_forecasting_metrics, calculate_residuals
from src.utils.seed import set_seed
from src.utils.device import print_device_info


def main():
    set_seed(42)
    print_device_info()

    dataset_path = PROJECT_ROOT / "data" / "raw" / "ETTh1.csv"

    df = load_ett_dataset(str(dataset_path))

    train_df, val_df, test_df = time_based_split(
        df=df,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
    )

    print_split_summary(train_df, val_df, test_df)

    scaled_data = prepare_scaled_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        input_type="multivariate",
        target_col="OT",
    )

    print_preprocessing_summary(scaled_data)

    windowed_data = prepare_windowed_dataloaders(
        scaled_data=scaled_data,
        input_len=96,
        horizon=24,
        batch_size=64,
    )

    print_window_summary(windowed_data)

    batch_X, batch_y = next(iter(windowed_data["train_loader"]))

    print("Batch check")
    print("-" * 40)
    print("batch_X shape:", batch_X.shape)
    print("batch_y shape:", batch_y.shape)
    print("-" * 40)

    # Dummy prediction for interface validation.
    # Here we use zeros with the same shape as test_y. This is not a real model
    # prediction; it only checks that evaluation can consume Module 1 outputs and
    # calculate metrics in the original OT scale.
    y_true_scaled = windowed_data["test_y"]
    y_pred_scaled = np.zeros_like(y_true_scaled)

    metrics = calculate_forecasting_metrics(
        y_true_scaled=y_true_scaled,
        y_pred_scaled=y_pred_scaled,
        scaler_y=windowed_data["scaler_y"],
    )

    residuals = calculate_residuals(
        y_true_scaled=y_true_scaled,
        y_pred_scaled=y_pred_scaled,
        scaler_y=windowed_data["scaler_y"],
        absolute=True,
    )

    print("Evaluation interface check")
    print("-" * 40)
    print("num_features:", windowed_data["num_features"])
    print("MAE:", metrics["mae"])
    print("RMSE:", metrics["rmse"])
    print("Residual shape:", residuals.shape)
    print("Residual min:", residuals.min())
    print("Residual max:", residuals.max())
    print("-" * 40)

    print("Module 1 interface check completed successfully.")


if __name__ == "__main__":
    main()
