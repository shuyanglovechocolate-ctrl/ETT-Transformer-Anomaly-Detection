"""Tests for anomaly diagnostics (Module 6.8)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    summarize_score_distribution, compute_causal_rolling_std,
    compute_flatness_score, compare_scores_by_label,
)


def test_summarize_score_distribution_values():
    df = pd.DataFrame({"anomaly_score": np.arange(0, 101, dtype=float)})
    s = summarize_score_distribution(df)
    assert s["median_score"] == 50.0
    assert s["p90"] == pytest.approx(90.0)
    assert s["max_score"] == 100.0
    assert s["num_points"] == 101


def test_summarize_empty_and_nan_and_missing_raise():
    with pytest.raises(ValueError):
        summarize_score_distribution(pd.DataFrame({"anomaly_score": []}))
    with pytest.raises(ValueError):
        summarize_score_distribution(pd.DataFrame({"anomaly_score": [1.0, np.nan]}))
    with pytest.raises(ValueError):
        summarize_score_distribution(pd.DataFrame({"other": [1.0]}))


def test_causal_rolling_std_does_not_use_current_or_future():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    base = compute_causal_rolling_std(y, window=10)
    y2 = y.copy()
    y2[50] += 100.0  # perturb one point
    perturbed = compute_causal_rolling_std(y2, window=10)
    # Score at index 50 uses [40..49], not index 50 itself -> unchanged.
    assert base.iloc[50] == pytest.approx(perturbed.iloc[50])
    # Earlier indices unchanged too (no future leakage).
    assert np.allclose(base.iloc[:50].dropna(), perturbed.iloc[:50].dropna())


def test_flatness_score_higher_on_flat_region():
    # First half variable, second half constant (flat).
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0, 1, 50), np.full(50, 5.0)])
    flat = compute_flatness_score(y, window=10)
    variable_region = flat.iloc[20:45].mean()
    flat_region = flat.iloc[70:95].mean()
    assert flat_region > variable_region


def test_compare_scores_by_label_ratio():
    df = pd.DataFrame({
        "is_anomaly": [True, True, False, False],
        "score_a": [10.0, 10.0, 1.0, 1.0],
    })
    cmp = compare_scores_by_label(df, ["score_a"], "is_anomaly")
    row = cmp.iloc[0]
    assert row["mean_anomaly"] == 10.0
    assert row["mean_normal"] == 1.0
    assert row["ratio_anomaly_to_normal"] == pytest.approx(10.0, rel=1e-4)


def test_compare_scores_missing_column_raises():
    df = pd.DataFrame({"is_anomaly": [True, False]})
    with pytest.raises(ValueError):
        compare_scores_by_label(df, ["missing"], "is_anomaly")
