"""Apply a threshold to anomaly scores to produce predicted labels (Module 6.3)."""

import numpy as np
import pandas as pd


def detect_anomalies(
    df: pd.DataFrame,
    threshold: float,
    score_col: str = "anomaly_score",
) -> pd.DataFrame:
    """Label points as anomalies where score exceeds the threshold.

    A strict ``>`` is used so a score exactly at the threshold is not flagged.
    The input dataframe is not modified; a copy with ``threshold`` and
    ``predicted_anomaly`` columns is returned.

    Parameters
    ----------
    df : pd.DataFrame
        Anomalous test dataframe containing ``score_col``.
    threshold : float
        Threshold learned from validation residuals; must be finite and >= 0.
    score_col : str
        Name of the anomaly score column.
    """
    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")

    scores = df[score_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite (no NaN/inf).")
    if np.any(scores < 0):
        raise ValueError("scores must be non-negative.")

    output = df.copy()
    output["threshold"] = float(threshold)
    output["predicted_anomaly"] = (output[score_col] > threshold).astype(bool)
    return output
