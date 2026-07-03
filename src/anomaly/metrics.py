"""Point-wise anomaly detection evaluation metrics (Module 6.4).

Compares synthetic anomaly labels (``is_anomaly``) with detector predictions
(``predicted_anomaly``) and reports the confusion matrix plus precision, recall,
F1, false-positive rate and true-negative rate. All ratios are zero-division
safe (a rate with an empty denominator is reported as 0.0).
"""

import numpy as np
import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _as_bool_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
    series = df[col]
    if series.isna().any():
        raise ValueError(f"Column '{col}' contains NaN.")
    return series.to_numpy().astype(bool)


def calculate_detection_metrics(
    df: pd.DataFrame,
    label_col: str = "is_anomaly",
    pred_col: str = "predicted_anomaly",
) -> dict:
    """Compute point-wise detection metrics from labels and predictions.

    Returns
    -------
    dict
        tp, fp, tn, fn, precision, recall, f1, false_positive_rate,
        true_negative_rate, num_points, num_true_anomaly, num_predicted_anomaly.
    """
    if len(df) == 0:
        raise ValueError("dataframe must not be empty.")

    labels = _as_bool_array(df, label_col)
    preds = _as_bool_array(df, pred_col)

    tp = int(np.sum(labels & preds))
    fp = int(np.sum(~labels & preds))
    tn = int(np.sum(~labels & ~preds))
    fn = int(np.sum(labels & ~preds))

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    fpr = _safe_ratio(fp, fp + tn)
    tnr = _safe_ratio(tn, tn + fp)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "true_negative_rate": tnr,
        "num_points": int(len(df)),
        "num_true_anomaly": int(np.sum(labels)),
        "num_predicted_anomaly": int(np.sum(preds)),
    }
