"""Detector score dispatch (shared by Modules 6.5-6.11).

Given an injected test dataframe and the clean validation residual dataframe,
returns ``(validation_scores, test_scores)`` for a detector. The residual
detector uses the aggregated residual anomaly score; the causal statistical
baselines score ``y_true`` (no forecasting model). Centralised here so the
detection, threshold-free, hybrid and cross-analysis experiments all reuse it.
"""

import numpy as np

from src.anomaly.baselines import compute_baseline_scores, BASELINE_DETECTORS
from src.anomaly.diagnostics import compute_flatness_score
from src.anomaly.hybrid import hybrid_rankmax_score

DETECTOR_TYPES = ("residual",) + BASELINE_DETECTORS + ("flatness", "hybrid_rankmax")
DEFAULT_ROLLING_WINDOW = 24
DEFAULT_FLATNESS_WINDOW = 12


def _flatness(series, window):
    # Leading positions without a full window have no flatness signal -> 0.
    return compute_flatness_score(series, window=window).fillna(0.0).to_numpy()


def score_detector(detector_type, injected_df, val_df,
                   rolling_window: int = DEFAULT_ROLLING_WINDOW,
                   flatness_window: int = DEFAULT_FLATNESS_WINDOW):
    """Return (validation_scores, test_scores) for a score-based detector.

    Supports: residual, the causal baselines, flatness, and hybrid_rankmax.
    (hybrid_or is a threshold OR rule, handled by the experiment runner.)
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

    if detector_type == "flatness":
        return (_flatness(val_df["y_true"], flatness_window),
                _flatness(injected_df["y_true_anomalous"], flatness_window))

    if detector_type == "hybrid_rankmax":
        val_residual = np.asarray(val_df["anomaly_score"], dtype=float)
        test_residual = np.asarray(injected_df["anomaly_score"], dtype=float)
        val_flatness = _flatness(val_df["y_true"], flatness_window)
        test_flatness = _flatness(injected_df["y_true_anomalous"], flatness_window)
        val_scores = hybrid_rankmax_score(val_residual, val_flatness,
                                          val_residual, val_flatness)
        test_scores = hybrid_rankmax_score(test_residual, test_flatness,
                                           val_residual, val_flatness)
        return val_scores, test_scores

    raise ValueError(
        f"Unknown detector_type: {detector_type}. "
        f"Expected one of {DETECTOR_TYPES}."
    )
