"""Plotting helpers for training and forecasting results."""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless saving
import matplotlib.pyplot as plt
import pandas as pd

from src.viz import PALETTE, apply_paper_style


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_loss_curve(history: dict, path: str, title: str = "Training Loss") -> None:
    """Save train/val loss curves.

    Markers are used so that single-epoch runs (one data point) are still
    visible. For parameter-free models (Naive) the history is empty, so an
    explanatory annotation is drawn instead of a blank plot.
    """
    _ensure_dir(path)
    apply_paper_style()
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    if not train_loss and not val_loss:
        plt.text(
            0.5, 0.5, "No training (parameter-free baseline)",
            ha="center", va="center", transform=plt.gca().transAxes,
        )
    else:
        if train_loss:
            plt.plot(range(1, len(train_loss) + 1), train_loss,
                     marker="o", label="train")
        if val_loss:
            plt.plot(range(1, len(val_loss) + 1), val_loss,
                     marker="o", label="val")
        plt.legend()

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (scaled MSE)")
    plt.tight_layout()
    plt.savefig(path)
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
    apply_paper_style()
    sub = df[df["horizon_index"] == horizon_index].sort_values("target_date")

    plt.figure(figsize=(14, 5))
    plt.plot(sub["target_date"], sub["y_true"], label="actual", color="#333333")
    plt.plot(sub["target_date"], sub["y_pred"], label="predicted",
             color=PALETTE[1], alpha=0.85)
    plt.title(f"{title} (horizon_index={horizon_index})")
    plt.xlabel("Date")
    plt.ylabel("Oil Temperature (OT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
