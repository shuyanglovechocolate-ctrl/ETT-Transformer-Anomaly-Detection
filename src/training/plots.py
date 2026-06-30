"""Plotting helpers for training and forecasting results."""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless saving
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_loss_curve(history: dict, path: str, title: str = "Training Loss") -> None:
    """Save train/val loss curves. No-op-safe for empty history (Naive)."""
    _ensure_dir(path)
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if history.get("train_loss"):
        plt.plot(epochs, history["train_loss"], label="train")
    if history.get("val_loss"):
        plt.plot(epochs, history["val_loss"], label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (scaled MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_prediction_plot(
    df: pd.DataFrame,
    path: str,
    horizon_index: int = 0,
    title: str = "Prediction vs Actual",
) -> None:
    """Plot one-step-ahead (or chosen horizon_index) true vs predicted over time.

    Selecting a single horizon_index gives a continuous, non-overlapping series.
    """
    _ensure_dir(path)
    sub = df[df["horizon_index"] == horizon_index].sort_values("target_date")

    plt.figure(figsize=(14, 5))
    plt.plot(sub["target_date"], sub["y_true"], label="actual")
    plt.plot(sub["target_date"], sub["y_pred"], label="predicted", alpha=0.8)
    plt.title(f"{title} (horizon_index={horizon_index})")
    plt.xlabel("Date")
    plt.ylabel("Oil Temperature (OT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
