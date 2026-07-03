"""Synthetic anomaly injection for anomaly detection (Module 6.2).

Operates on an aggregated residual/prediction series (one row per target_date,
from Module 6.1). Anomalies are injected into ``y_true`` only; ``y_pred`` (the
model's clean prediction of normal behaviour) is left unchanged, so the
anomalous residual ``y_true_anomalous - y_pred`` grows where anomalies occur.

Supported types:
- spike       : additive offset over a short segment.
- level_shift : additive offset over a longer segment.
- frozen      : the value is held constant (sensor stuck).

Segments are non-overlapping and reproducible for a given seed.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

ANOMALY_TYPES = ("spike", "level_shift", "frozen")
_REQUIRED_COLUMNS = {"target_date", "y_true", "y_pred"}


def _validate(anomaly_type, anomaly_ratio, duration_range):
    if anomaly_type not in ANOMALY_TYPES:
        raise ValueError(f"anomaly_type must be one of {ANOMALY_TYPES}, got '{anomaly_type}'.")
    if not 0.0 < anomaly_ratio < 1.0:
        raise ValueError(f"anomaly_ratio must be in (0, 1), got {anomaly_ratio}.")
    lo, hi = duration_range
    if not (isinstance(lo, int) and isinstance(hi, int)):
        raise ValueError("duration_range must be integers.")
    if lo < 1 or hi < lo:
        raise ValueError(f"duration_range must satisfy 1 <= lo <= hi, got {duration_range}.")


def _place_segments(n, anomaly_ratio, num_segments, duration_range, rng):
    """Place non-overlapping (start, end, segment_id) segments."""
    occupied = np.zeros(n, dtype=bool)
    segments = []
    target_points = max(1, round(anomaly_ratio * n))
    placed = 0
    seg_id = 0
    attempts = 0
    max_attempts = 2000

    def enough():
        if num_segments is not None:
            return seg_id >= num_segments
        return placed >= target_points

    while not enough() and attempts < max_attempts:
        attempts += 1
        d = int(rng.integers(duration_range[0], duration_range[1] + 1))
        d = min(d, n)
        start = int(rng.integers(0, n - d + 1))
        end = start + d
        if occupied[start:end].any():
            continue
        occupied[start:end] = True
        seg_id += 1
        segments.append((start, end, seg_id))
        placed += d
    return segments


def inject_synthetic_anomalies(
    df: pd.DataFrame,
    anomaly_type: str,
    anomaly_ratio: float = 0.02,
    num_segments: Optional[int] = None,
    duration_range: Tuple[int, int] = (3, 12),
    magnitude_scale: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject labelled synthetic anomalies into an aggregated residual series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain target_date, y_true, y_pred (e.g. Module 6.1 output).
    anomaly_type : str
        One of 'spike', 'level_shift', 'frozen'.
    anomaly_ratio : float
        Approximate fraction of timestamps to make anomalous (ignored if
        num_segments is given).
    num_segments : Optional[int]
        Fixed number of anomaly segments (overrides anomaly_ratio when set).
    duration_range : Tuple[int, int]
        Inclusive range of segment lengths in timesteps.
    magnitude_scale : float
        Offset magnitude in units of std(y_true) (used by spike / level_shift).
    seed : int
        Reproducibility seed for segment placement and offset signs.

    Returns
    -------
    pd.DataFrame
        Columns: target_date, y_true_original, y_true_anomalous, y_pred,
        residual_anomalous, abs_residual_anomalous, anomaly_score, is_anomaly,
        anomaly_type, anomaly_segment_id (+ aggregation_method if present).
    """
    _validate(anomaly_type, anomaly_ratio, duration_range)
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"input dataframe missing columns: {sorted(missing)}")

    df = df.reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(seed)

    y_orig = df["y_true"].to_numpy(dtype=float)
    y_anom = y_orig.copy()
    y_pred = df["y_pred"].to_numpy(dtype=float)

    std = float(np.std(y_orig))
    if std == 0.0:
        std = 1.0
    magnitude = magnitude_scale * std

    is_anomaly = np.zeros(n, dtype=bool)
    segment_id = np.zeros(n, dtype=int)

    for start, end, sid in _place_segments(n, anomaly_ratio, num_segments, duration_range, rng):
        if anomaly_type in ("spike", "level_shift"):
            sign = 1.0 if rng.random() < 0.5 else -1.0
            y_anom[start:end] = y_orig[start:end] + sign * magnitude
        elif anomaly_type == "frozen":
            y_anom[start:end] = y_orig[start]
        is_anomaly[start:end] = True
        segment_id[start:end] = sid

    residual_anom = y_anom - y_pred
    out = pd.DataFrame({
        "target_date": df["target_date"].to_numpy(),
        "y_true_original": y_orig,
        "y_true_anomalous": y_anom,
        "y_pred": y_pred,
        "residual_anomalous": residual_anom,
        "abs_residual_anomalous": np.abs(residual_anom),
        "anomaly_score": np.abs(residual_anom),
        "is_anomaly": is_anomaly,
        "anomaly_type": np.where(is_anomaly, anomaly_type, ""),
        "anomaly_segment_id": segment_id,
    })
    if "aggregation_method" in df.columns:
        out["aggregation_method"] = df["aggregation_method"].to_numpy()
    return out
