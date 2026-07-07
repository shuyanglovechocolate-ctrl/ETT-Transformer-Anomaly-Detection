"""Tests for synthetic anomaly injection (Module 6.2)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import inject_synthetic_anomalies, ANOMALY_TYPES


def _series(n=200):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=n, freq="1h").astype(str)
    return pd.DataFrame({
        "target_date": dates,
        "y_true": rng.normal(10.0, 2.0, size=n),
        "y_pred": rng.normal(10.0, 2.0, size=n),
        "aggregation_method": "first",
    })


def test_spike_changes_y_true_at_anomalies():
    out = inject_synthetic_anomalies(_series(), "spike", anomaly_ratio=0.05, seed=1)
    anom = out[out.is_anomaly]
    assert len(anom) > 0
    assert (anom["y_true_anomalous"] != anom["y_true_original"]).all()


def test_level_shift_constant_offset_per_segment():
    out = inject_synthetic_anomalies(_series(), "level_shift",
                                     duration_range=(12, 24), anomaly_ratio=0.1, seed=2)
    for sid, g in out[out.is_anomaly].groupby("anomaly_segment_id"):
        offset = g["y_true_anomalous"] - g["y_true_original"]
        assert np.allclose(offset, offset.iloc[0])  # constant within segment


def test_frozen_holds_constant_value():
    out = inject_synthetic_anomalies(_series(), "frozen",
                                     duration_range=(12, 24), anomaly_ratio=0.1, seed=3)
    for sid, g in out[out.is_anomaly].groupby("anomaly_segment_id"):
        assert g["y_true_anomalous"].nunique() == 1  # held constant


def test_y_pred_unchanged():
    src = _series()
    out = inject_synthetic_anomalies(src, "spike", anomaly_ratio=0.05, seed=4)
    assert np.array_equal(out["y_pred"].to_numpy(), src["y_pred"].to_numpy())


def test_labels_and_segment_ids_consistent():
    out = inject_synthetic_anomalies(_series(), "spike", anomaly_ratio=0.05, seed=5)
    # segment_id > 0 exactly where is_anomaly is True
    assert ((out["anomaly_segment_id"] > 0) == out["is_anomaly"]).all()
    # anomaly_score equals |residual_anomalous|
    assert np.allclose(out["anomaly_score"], out["abs_residual_anomalous"])


def test_anomalous_residual_recomputed():
    out = inject_synthetic_anomalies(_series(), "spike", anomaly_ratio=0.05, seed=6)
    expected = out["y_true_anomalous"] - out["y_pred"]
    assert np.allclose(out["residual_anomalous"], expected)


def test_same_seed_reproducible_diff_seed_differs():
    a = inject_synthetic_anomalies(_series(), "spike", anomaly_ratio=0.05, seed=7)
    b = inject_synthetic_anomalies(_series(), "spike", anomaly_ratio=0.05, seed=7)
    c = inject_synthetic_anomalies(_series(), "spike", anomaly_ratio=0.05, seed=99)
    assert np.array_equal(a["is_anomaly"].to_numpy(), b["is_anomaly"].to_numpy())
    assert not np.array_equal(a["is_anomaly"].to_numpy(), c["is_anomaly"].to_numpy())


def test_segments_do_not_overlap_and_stay_in_bounds():
    out = inject_synthetic_anomalies(_series(300), "level_shift",
                                     duration_range=(5, 10), anomaly_ratio=0.1, seed=8)
    # each anomalous position belongs to exactly one segment; bounds valid
    assert out["anomaly_segment_id"].max() >= 1
    assert len(out) == 300
    assert out["is_anomaly"].sum() > 0


# ---------------------------------------------------------------------------
# Extended, more fault-like anomaly types (Module 7.7)
# ---------------------------------------------------------------------------

def test_extended_types_registered():
    for t in ("drift", "noise_burst", "stuck_with_jitter"):
        assert t in ANOMALY_TYPES


def test_drift_is_monotonic_ramp_from_zero():
    out = inject_synthetic_anomalies(_series(), "drift",
                                     duration_range=(12, 24), anomaly_ratio=0.1, seed=11)
    segs = out[out.is_anomaly].groupby("anomaly_segment_id")
    assert segs.ngroups > 0
    for _, g in segs:
        offset = np.abs((g["y_true_anomalous"] - g["y_true_original"]).to_numpy())
        assert offset[0] == pytest.approx(0.0, abs=1e-9)   # ramp starts at ~0
        assert np.all(np.diff(offset) >= -1e-9)            # non-decreasing
        assert offset[-1] > offset[0]                      # grows across segment


def test_noise_burst_adds_variable_noise():
    out = inject_synthetic_anomalies(_series(), "noise_burst",
                                     duration_range=(12, 24), anomaly_ratio=0.1, seed=12)
    for _, g in out[out.is_anomaly].groupby("anomaly_segment_id"):
        offset = (g["y_true_anomalous"] - g["y_true_original"]).to_numpy()
        if len(offset) > 3:
            assert np.std(offset) > 0                      # not a constant offset


def test_stuck_with_jitter_near_constant_but_not_frozen():
    out = inject_synthetic_anomalies(_series(), "stuck_with_jitter",
                                     duration_range=(12, 24), anomaly_ratio=0.1,
                                     jitter_scale=0.1, seed=13)
    global_std = _series()["y_true"].std()
    for _, g in out[out.is_anomaly].groupby("anomaly_segment_id"):
        vals = g["y_true_anomalous"].to_numpy()
        if len(vals) > 3:
            assert vals.std() > 0                          # jitter -> not exactly frozen
            assert vals.std() < global_std                 # but much flatter than the signal


def test_invalid_anomaly_type_raises():
    with pytest.raises(ValueError):
        inject_synthetic_anomalies(_series(), "outlier")


def test_invalid_ratio_and_duration_raise():
    with pytest.raises(ValueError):
        inject_synthetic_anomalies(_series(), "spike", anomaly_ratio=1.5)
    with pytest.raises(ValueError):
        inject_synthetic_anomalies(_series(), "spike", duration_range=(10, 3))
