"""Tests for anomaly-detection significance analysis (Module 7.4)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.analyze_anomaly_significance import (
    matched_metric_frame,
    paired_bootstrap_ci,
    wilcoxon_greater_pvalue,
    compute_significance,
    PAIR_KEY,
)


def _row(model, seed, detector, pr_auc, best_f1, anomaly_type="spike"):
    return {
        "dataset": "ETTh1", "model": model, "horizon": 24,
        "aggregation_method": "first", "anomaly_type": anomaly_type,
        "injection_seed": seed, "detector_type": detector,
        "pr_auc": pr_auc, "best_f1": best_f1,
    }


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Matched (paired) units
# ---------------------------------------------------------------------------

def test_matched_frame_keeps_only_complete_cases():
    rows = [
        _row("nlinear", 42, "residual", 0.9, 0.8),
        _row("nlinear", 42, "raw_zscore", 0.5, 0.4),   # unit A: complete
        _row("nlinear", 2024, "residual", 0.7, 0.6),   # unit B: baseline missing
    ]
    wide = matched_metric_frame(_frame(rows), "pr_auc",
                                ["residual", "raw_zscore"])
    assert len(wide) == 1  # only the complete unit survives
    assert set(wide.columns) == {"residual", "raw_zscore"}
    assert wide["residual"].iloc[0] == 0.9


def test_matched_frame_drops_nan_metric():
    rows = [
        _row("nlinear", 42, "residual", 0.9, 0.8),
        _row("nlinear", 42, "raw_zscore", np.nan, 0.4),  # NaN metric -> drop unit
    ]
    wide = matched_metric_frame(_frame(rows), "pr_auc",
                                ["residual", "raw_zscore"])
    assert len(wide) == 0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_positive_diffs_are_significant():
    diffs = np.full(36, 0.2)
    res = paired_bootstrap_ci(diffs, n_boot=2000, seed=42)
    assert res["significant"] is True
    assert res["ci_low"] > 0
    assert res["win_rate"] == 1.0
    assert res["num_pairs"] == 36
    assert res["mean_diff"] == np.mean(diffs)


def test_bootstrap_negative_diffs_not_significant():
    diffs = np.full(36, -0.2)
    res = paired_bootstrap_ci(diffs, n_boot=2000, seed=42)
    assert res["significant"] is False
    assert res["ci_high"] < 0
    assert res["win_rate"] == 0.0


def test_bootstrap_empty_is_safe():
    res = paired_bootstrap_ci(np.array([]), n_boot=100, seed=42)
    assert res["num_pairs"] == 0
    assert res["significant"] is False
    assert np.isnan(res["mean_diff"])


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.1, 0.05, size=36)
    a = paired_bootstrap_ci(diffs, n_boot=1000, seed=7)
    b = paired_bootstrap_ci(diffs, n_boot=1000, seed=7)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------

def test_wilcoxon_positive_diffs_small_pvalue():
    diffs = np.linspace(0.05, 0.4, 20)  # all positive
    p = wilcoxon_greater_pvalue(diffs)
    assert not np.isnan(p)
    assert p < 0.05


def test_wilcoxon_all_zero_is_nan():
    assert np.isnan(wilcoxon_greater_pvalue(np.zeros(10)))
    assert np.isnan(wilcoxon_greater_pvalue(np.array([])))


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_compute_significance_end_to_end():
    rows = []
    for seed in (42, 2024, 3407):
        # residual clearly beats the baseline on both metrics
        rows.append(_row("nlinear", seed, "residual", 0.95, 0.9))
        rows.append(_row("nlinear", seed, "raw_zscore", 0.5, 0.45))
    out = compute_significance(_frame(rows), n_boot=1000, seed=42)

    expected_cols = ["anomaly_type", "baseline_detector", "metric", "num_pairs",
                     "mean_residual", "mean_baseline", "mean_diff", "ci_low",
                     "ci_high", "significant", "wilcoxon_pvalue", "win_rate",
                     "n_boot", "bootstrap_seed"]
    assert list(out.columns) == expected_cols

    # only raw_zscore has matched units; the other baselines yield empty rows
    raw = out[out["baseline_detector"] == "raw_zscore"]
    assert set(raw["metric"]) == {"pr_auc", "oracle_best_f1"}
    for _, r in raw.iterrows():
        assert r["num_pairs"] == 3
        assert r["significant"] == (r["ci_low"] > 0)
        assert r["mean_diff"] > 0
        assert r["win_rate"] == 1.0

    missing = out[out["baseline_detector"] == "diff_score"]
    assert (missing["num_pairs"] == 0).all()
