"""Tests for the Module 3.5 result-analysis and validation scripts."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.analyze_forecasting_results import (
    per_horizon_metrics_for_experiment,
    summarize_per_horizon,
    best_model_by_dataset_horizon,
    bootstrap_mae_diff_ci,
)
from experiments.validate_results import validate


def _pred_df():
    # horizon_index 0: errors [1,1]; horizon_index 1: errors [2,2].
    return pd.DataFrame({
        "sample_index": [0, 1, 0, 1],
        "horizon_index": [0, 0, 1, 1],
        "y_true": [10.0, 10.0, 10.0, 10.0],
        "y_pred": [9.0, 9.0, 8.0, 8.0],
        "residual": [1.0, 1.0, 2.0, 2.0],
        "abs_residual": [1.0, 1.0, 2.0, 2.0],
    })


def test_per_horizon_metrics_for_experiment():
    ph = per_horizon_metrics_for_experiment(_pred_df())
    ph = ph.set_index("horizon_index")
    assert ph.loc[0, "mae"] == 1.0
    assert ph.loc[1, "mae"] == 2.0
    # error grows with horizon step
    assert ph.loc[1, "mae"] > ph.loc[0, "mae"]


def test_summarize_per_horizon_aggregates_seeds():
    base = per_horizon_metrics_for_experiment(_pred_df())
    frames = []
    for seed in (42, 2024, 3407):
        f = base.copy()
        f.insert(0, "seed", seed)
        for col, val in [("horizon", 24), ("input_type", "multivariate"),
                         ("model", "linear"), ("dataset", "ETTh1")]:
            f.insert(0, col, val)
        frames.append(f)
    summary = summarize_per_horizon(pd.concat(frames, ignore_index=True))
    assert (summary["num_runs"] == 3).all()


def test_bootstrap_mae_diff_ci_detects_difference():
    a = np.ones(500) * 1.0
    b = np.ones(500) * 2.0
    res = bootstrap_mae_diff_ci(a, b, n_boot=200, seed=0)
    assert res["mae_diff_a_minus_b"] == -1.0
    assert res["a_better"] is True
    assert res["significant"] is True


def test_bootstrap_identical_not_significant():
    a = np.random.default_rng(0).normal(size=500)
    res = bootstrap_mae_diff_ci(a, a, n_boot=200, seed=0)
    assert res["mae_diff_a_minus_b"] == 0.0
    assert res["significant"] is False


def test_best_model_by_dataset_horizon_picks_min_mae():
    log = pd.DataFrame({
        "dataset": ["ETTh1"] * 6,
        "model": ["linear", "linear", "linear", "lstm", "lstm", "lstm"],
        "input_type": ["multivariate"] * 6,
        "horizon": [24] * 6,
        "seed": [42, 2024, 3407] * 2,
        "mae": [1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
        "rmse": [1.0] * 6, "wape": [10.0] * 6,
    })
    best = best_model_by_dataset_horizon(log)
    assert len(best) == 1
    assert best.iloc[0]["best_model"] == "linear"


def test_best_model_excludes_regularized():
    log = pd.DataFrame({
        "dataset": ["ETTh1"] * 2,
        "model": ["transformer_regularized", "linear"],
        "input_type": ["multivariate"] * 2,
        "horizon": [96] * 2,
        "seed": [42, 42],
        "mae": [0.5, 1.0], "rmse": [1.0, 1.0], "wape": [10.0, 10.0],
    })
    best = best_model_by_dataset_horizon(log)
    # Even though the regularized variant has lower mae, it is excluded.
    assert best.iloc[0]["best_model"] == "linear"


def test_validate_missing_log(tmp_path):
    report = validate(str(tmp_path))
    assert report["ok"] is False
    assert report["checks"][0]["check"] == "experiment_log_exists"
    assert report["checks"][0]["ok"] is False
