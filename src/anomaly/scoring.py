"""Detector score dispatch (shared by Modules 6.5-6.11).

Given an injected test dataframe and the clean validation residual dataframe,
returns ``(validation_scores, test_scores)`` for a detector. The residual
detector uses the aggregated residual anomaly score; the causal statistical
baselines score ``y_true`` (no forecasting model). Centralised here so the
detection, threshold-free, hybrid and cross-analysis experiments all reuse it.
"""

import numpy as np

from src.anomaly.baselines import compute_baseline_scores, BASELINE_DETECTORS

DETECTOR_TYPES = ("residual",) + BASELINE_DETECTORS
DEFAULT_ROLLING_WINDOW = 24


def score_detector(detector_type, injected_df, val_df,
                   rolling_window: int = DEFAULT_ROLLING_WINDOW):
    """Return (validation_scores, test_scores) for a detector.

    Parameters
    ----------
    detector_type : str
        "residual" or a baseline in BASELINE_DETECTORS.
    injected_df : pd.DataFrame
        Test dataframe with injected anomalies (has anomaly_score and
        y_true_anomalous).
    val_df : pd.DataFrame
        Clean validation residual dataframe (has anomaly_score and y_true).
    rolling_window : int
        Window for the rolling_zscore baseline.
    """
    if detector_type == "residual":
        return (np.asarray(val_df["anomaly_score"], dtype=float),
                np.asarray(injected_df["anomaly_score"], dtype=float))

    if detector_type in BASELINE_DETECTORS:
        ref = val_df["y_true"].to_numpy()
        val_scores = compute_baseline_scores(
            detector_type, ref, ref_series=ref, window=rolling_window)
        test_scores = compute_baseline_scores(
            detector_type, injected_df["y_true_anomalous"].to_numpy(),
            ref_series=ref, window=rolling_window)
        return val_scores, test_scores

    raise ValueError(
        f"Unknown detector_type: {detector_type}. "
        f"Expected one of {DETECTOR_TYPES}."
    )
