"""Hybrid residual + flatness anomaly detection (Module 6.10).

The residual detector is strong for magnitude anomalies but weak for frozen /
stuck-sensor faults, which are better characterised by temporal flatness
(Module 6.8). This module combines the two complementary signals:

- ``hybrid_or``      : residual_alarm OR flatness_alarm (each threshold from
                       validation). A deployable, easy-to-explain rule.
- ``hybrid_rankmax`` : max of the two scores' validation-percentile ranks. A
                       scale-free continuous score for threshold-free (PR-AUC)
                       comparison. Uses max (not mean) so a single strong signal
                       is not diluted.

All references (rank normalization, thresholds) come only from validation.
"""

import numpy as np


def rank_normalize(scores, reference_scores) -> np.ndarray:
    """Percentile rank of each score within the reference distribution -> [0, 1].

    Depends only on ``reference_scores`` (from validation), never on the scores
    being ranked, so it is leakage-free.
    """
    ref = np.sort(np.asarray(reference_scores, dtype=float))
    s = np.asarray(scores, dtype=float)
    if len(ref) == 0:
        raise ValueError("reference_scores must not be empty.")
    return np.searchsorted(ref, s, side="right") / len(ref)


def hybrid_rankmax_score(residual_scores, flatness_scores,
                         val_residual_scores, val_flatness_scores) -> np.ndarray:
    """max( rank(residual | val_residual), rank(flatness | val_flatness) )."""
    residual_rank = rank_normalize(residual_scores, val_residual_scores)
    flatness_rank = rank_normalize(flatness_scores, val_flatness_scores)
    return np.maximum(residual_rank, flatness_rank)


def detect_hybrid_or(residual_scores, flatness_scores,
                     residual_threshold, flatness_threshold) -> np.ndarray:
    """Boolean alarms: residual above its threshold OR flatness above its own."""
    residual_alarm = np.asarray(residual_scores, dtype=float) > residual_threshold
    flatness_alarm = np.asarray(flatness_scores, dtype=float) > flatness_threshold
    return residual_alarm | flatness_alarm
