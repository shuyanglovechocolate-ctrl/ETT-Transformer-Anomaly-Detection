"""Anomaly detection visualisation (Module 6.5)."""

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd


def plot_anomaly_detection(df: pd.DataFrame, path: str, title: str = "") -> None:
    """Two-panel anomaly figure.

    Top: clean OT, anomalous OT, prediction, with true-anomaly points marked.
    Bottom: anomaly score, threshold line, and detected points.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dates = pd.to_datetime(df["target_date"])
    is_anom = df["is_anomaly"].to_numpy(dtype=bool)
    pred = df["predicted_anomaly"].to_numpy(dtype=bool)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(dates, df["y_true_original"], label="OT (clean)", alpha=0.6)
    ax1.plot(dates, df["y_true_anomalous"], label="OT (with anomalies)", alpha=0.85)
    ax1.plot(dates, df["y_pred"], label="prediction", alpha=0.6)
    ax1.scatter(dates[is_anom], df["y_true_anomalous"].to_numpy()[is_anom],
                color="red", s=12, zorder=5, label="true anomaly")
    ax1.set_ylabel("Oil Temperature (OT)")
    ax1.set_title(title)
    ax1.legend(loc="upper right")

    ax2.plot(dates, df["anomaly_score"], color="tab:blue", label="anomaly score")
    ax2.axhline(float(df["threshold"].iloc[0]), color="green", linestyle="--",
                label="threshold")
    ax2.scatter(dates[pred], df["anomaly_score"].to_numpy()[pred],
                color="orange", s=12, zorder=5, label="detected")
    ax2.set_ylabel("anomaly score")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
