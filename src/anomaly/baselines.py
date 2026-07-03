"""Simple causal statistical anomaly baselines (Module 6.6).

These baselines use only the observed target series (no forecasting model), so
they test whether forecast residuals add value over trivial statistics. All are
causal (they never use future values), and their thresholds are learned from
validation scores using the same protocol as the residual detector.

- raw_zscore    : |y_t - mean(val_y)| / std(val_y)      (deviation from normal level)
- diff_score    : |y_t - y_{t-1}|                        (abrupt change)
- rolling_zscore: |y_t - past_mean| / past_std           (deviation from recent past)
"""

import numpy as np
import pandas as pd

BASELINE_DETECTORS = ("raw_zscore", "diff_score", "rolling_zscore")
_EPS = 1e-8


def compute_baseline_scores(
    detector_type: str,
    series,
    ref_series=None,
    window: int = 24,
) -> np.ndarray:
    """Compute non-negative, causal anomaly scores for a baseline detector.

    Parameters
    ----------
    detector_type : str
        One of BASELINE_DETECTORS.
    series : array-like
        The series to score (validation y or anomalous test y).
    ref_series : array-like, optional
        Validation reference series (used by raw_zscore for mean/std).
    window : int
        Causal window length for rolling_zscore.
    """
    series = np.asarray(series, dtype=float)

    if detector_type == "raw_zscore":
        if ref_series is None:
            raise ValueError("raw_zscore requires ref_series (validation y).")
        ref = np.asarray(ref_series, dtype=float)
        return np.abs(series - ref.mean()) / (ref.std(ddof=0) + _EPS)

    if detector_type == "diff_score":
        # |y_t - y_{t-1}|; first point has no predecessor -> 0.
        return np.abs(np.diff(series, prepend=series[0]))

    if detector_type == "rolling_zscore":
        s = pd.Series(series)
        # shift(1) -> the window at t covers [t-window, t-1] only (causal).
        past_mean = s.shift(1).rolling(window, min_periods=1).mean()
        past_std = s.shift(1).rolling(window, min_periods=1).std(ddof=0)
        score = (s - past_mean).abs() / (past_std + _EPS)
        return score.fillna(0.0).to_numpy()

    raise ValueError(
        f"Unknown baseline detector: {detector_type}. "
        f"Expected one of {BASELINE_DETECTORS}."
    )
