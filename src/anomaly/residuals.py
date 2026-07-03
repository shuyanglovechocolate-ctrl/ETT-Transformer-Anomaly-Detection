"""Residual preparation and aggregation for anomaly detection (Module 6.1).

A prediction file is long-format (one row per sample x horizon step), so a single
target_date can be predicted by several windows. Anomaly detection needs one
score per timestamp, so residuals are aggregated per target_date.

``aggregate_residuals`` recomputes the residual from ``y_true`` and ``y_pred``
(rather than trusting a precomputed column), so the SAME function works for both
clean residuals and post-injection anomalous residuals (where only ``y_true``
changes). This keeps the clean-vs-anomalous pipelines consistent.
"""

import os
import time

import pandas as pd

AGGREGATION_METHODS = ("first", "mean", "max")
_REQUIRED_COLUMNS = {"target_date", "horizon_index", "y_true", "y_pred"}


def load_prediction_file(path: str, retries: int = 3, delay: float = 1.0) -> pd.DataFrame:
    """Read a prediction CSV, retrying on transient filesystem errors."""
    last_exc = None
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc


def aggregate_residuals(df: pd.DataFrame, method: str = "first") -> pd.DataFrame:
    """Aggregate a long-format prediction dataframe to one row per target_date.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain target_date, horizon_index, y_true, y_pred.
    method : str
        - "first": keep only horizon_index == 0 (one-step-ahead residual).
        - "mean" : per target_date, anomaly_score = mean(abs_residual).
        - "max"  : per target_date, anomaly_score = max(abs_residual).

    Returns
    -------
    pd.DataFrame
        Columns: target_date, y_true, y_pred, residual, abs_residual,
        anomaly_score, aggregation_method. One unique row per target_date.
    """
    if method not in AGGREGATION_METHODS:
        raise ValueError(
            f"method must be one of {AGGREGATION_METHODS}, got '{method}'."
        )
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"prediction dataframe missing columns: {sorted(missing)}")

    df = df.copy()
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_residual"] = df["residual"].abs()

    if method == "first":
        out = df[df["horizon_index"] == 0].copy()
        out["anomaly_score"] = out["abs_residual"]
        out = out[["target_date", "y_true", "y_pred", "residual", "abs_residual",
                   "anomaly_score"]]
    else:
        grouped = df.groupby("target_date")
        out = grouped.agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            anomaly_score=("abs_residual", method),
        ).reset_index()
        out["residual"] = out["y_true"] - out["y_pred"]
        out["abs_residual"] = out["residual"].abs()
        out = out[["target_date", "y_true", "y_pred", "residual", "abs_residual",
                   "anomaly_score"]]

    out["aggregation_method"] = method
    out = out.sort_values("target_date").reset_index(drop=True)
    return out


def save_aggregated_residuals(df: pd.DataFrame, output_path: str) -> None:
    """Save an aggregated-residual dataframe to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
