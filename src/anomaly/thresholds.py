"""Anomaly thresholds learned from validation residual scores (Module 6.3).

Thresholds are computed ONLY from validation residual scores, never from the
test set or anomaly labels, so detection stays leakage-free. Residual anomaly
scores are absolute errors, so they must be finite and non-negative.
"""

import numpy as np

THRESHOLD_METHODS = ("percentile", "mean_std", "iqr", "mad")


def validate_scores(scores) -> np.ndarray:
    """Return scores as a float array, rejecting empty/NaN/inf/negative input."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("scores must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must be finite (no NaN/inf).")
    if np.any(arr < 0):
        raise ValueError("scores must be non-negative (they are absolute errors).")
    return arr


def compute_threshold(scores, method: str = "percentile", **kwargs) -> float:
    """Compute an anomaly threshold from validation scores.

    Methods:
    - percentile : np.percentile(scores, percentile), default percentile=99.
    - mean_std   : mean + k*std, default k=3.0.
    - iqr        : Q3 + k*IQR, default k=1.5.
    - mad        : median + k*1.4826*MAD (robust), default k=3.5.
    """
    arr = validate_scores(scores)

    if method == "percentile":
        percentile = kwargs.get("percentile", 99)
        if not 0 < percentile < 100:
            raise ValueError("percentile must be in (0, 100).")
        return float(np.percentile(arr, percentile))

    if method == "mean_std":
        k = kwargs.get("k", 3.0)
        if k <= 0:
            raise ValueError("k must be positive.")
        return float(arr.mean() + k * arr.std(ddof=0))

    if method == "iqr":
        k = kwargs.get("k", 1.5)
        if k <= 0:
            raise ValueError("k must be positive.")
        q1, q3 = np.percentile(arr, [25, 75])
        return float(q3 + k * (q3 - q1))

    if method == "mad":
        k = kwargs.get("k", 3.5)
        if k <= 0:
            raise ValueError("k must be positive.")
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        return float(median + k * 1.4826 * mad)

    raise ValueError(
        f"Unsupported threshold method: {method}. "
        f"Expected one of {THRESHOLD_METHODS}."
    )
