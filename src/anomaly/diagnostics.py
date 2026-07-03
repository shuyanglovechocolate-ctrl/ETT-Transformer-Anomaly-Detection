"""Diagnostic tools for anomaly detection (Module 6.8).

Diagnostic-only: these functions explain WHY the residual detector behaves as it
does (score distribution shape, autocorrelation) and why it is weak on
frozen-value anomalies (residual magnitude separates them poorly, whereas a
causal temporal-flatness feature separates them clearly). They do not define a
new detector.
"""

import numpy as np
import pandas as pd

_EPS = 1e-6


def summarize_score_distribution(df: pd.DataFrame, score_col: str = "anomaly_score") -> dict:
    """Distribution statistics of an anomaly-score column."""
    if score_col not in df.columns:
        raise ValueError(f"Missing column: {score_col}")
    if len(df) == 0:
        raise ValueError("dataframe must not be empty.")
    s = df[score_col]
    if s.isna().any():
        raise ValueError(f"Column '{score_col}' contains NaN.")
    return {
        "num_points": int(len(s)),
        "mean_score": float(s.mean()),
        "std_score": float(s.std(ddof=0)),
        "median_score": float(s.median()),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max_score": float(s.max()),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurt()),
        "lag1_autocorrelation": float(s.autocorr(lag=1)),
    }


def compute_causal_rolling_std(series, window: int = 12) -> pd.Series:
    """Causal rolling standard deviation: uses only [t-window, t-1] at each t."""
    s = pd.Series(np.asarray(series, dtype=float))
    return s.shift(1).rolling(window=window, min_periods=window).std(ddof=0)


def compute_flatness_score(series, window: int = 12, epsilon: float = _EPS) -> pd.Series:
    """Flatness score = 1 / (causal rolling std + epsilon).

    Larger when the recent past is unusually flat (e.g. a stuck/frozen sensor).
    Leading positions without a full window are NaN.
    """
    rolling_std = compute_causal_rolling_std(series, window=window)
    return 1.0 / (rolling_std + epsilon)


def compare_scores_by_label(df: pd.DataFrame, score_cols, label_col: str = "is_anomaly") -> pd.DataFrame:
    """Compare each score column's anomaly vs normal separation."""
    if label_col not in df.columns:
        raise ValueError(f"Missing column: {label_col}")
    labels = df[label_col].to_numpy().astype(bool)
    rows = []
    for col in score_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        s = df[col]
        anom = s[labels]
        normal = s[~labels]
        mean_anom = float(anom.mean())
        mean_normal = float(normal.mean())
        rows.append({
            "score_col": col,
            "mean_anomaly": mean_anom,
            "mean_normal": mean_normal,
            "ratio_anomaly_to_normal": mean_anom / (mean_normal + _EPS),
            "median_anomaly": float(anom.median()),
            "median_normal": float(normal.median()),
        })
    return pd.DataFrame(rows)
