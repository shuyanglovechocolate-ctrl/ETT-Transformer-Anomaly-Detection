"""Tests for the ETTm external-validity analysis (minute-level frequency effect)."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.analyze_ettm_external_validity import (
    build_comparison, verdict, OUTPUT_COLS,
)


def _row(dataset, model, mae, params, horizon=96):
    return {
        "dataset": dataset, "model": model, "input_type": "multivariate",
        "horizon": horizon, "seed": 42, "mae": mae, "rmse": mae * 1.3,
        "wape": mae * 10, "total_parameters": params,
    }


def _log(rows):
    return pd.DataFrame(rows)


def test_comparison_ranks_by_mae_within_dataset():
    df = _log([
        _row("ETTm1", "nlinear", 0.40, 16000),
        _row("ETTm1", "transformer", 0.55, 69000),
        _row("ETTm1", "naive", 0.80, 0),
    ])
    out = build_comparison(df)
    assert list(out.columns) == OUTPUT_COLS
    m1 = out[out["dataset"] == "ETTm1"].sort_values("rank_within_dataset")
    assert list(m1["model"]) == ["nlinear", "transformer", "naive"]
    assert list(m1["rank_within_dataset"]) == [1, 2, 3]


def test_verdict_flags_linear_beating_transformer():
    df = _log([
        _row("ETTm1", "dlinear", 0.40, 32000),
        _row("ETTm1", "nlinear", 0.42, 16000),
        _row("ETTm1", "transformer", 0.55, 69000),
        _row("ETTm2", "nlinear", 0.60, 16000),
        _row("ETTm2", "transformer", 0.50, 69000),  # transformer wins here
    ])
    v = verdict(build_comparison(df)).set_index("dataset")
    assert v.loc["ETTm1", "best_linear_model"] == "dlinear"
    assert bool(v.loc["ETTm1", "linear_beats_transformer"]) is True
    assert bool(v.loc["ETTm2", "linear_beats_transformer"]) is False


def test_stray_header_row_dropped():
    df = _log([_row("ETTm1", "linear", 0.5, 16000)])
    header = {c: c for c in df.columns}
    df = pd.concat([df, pd.DataFrame([header])], ignore_index=True)
    out = build_comparison(df)
    assert set(out["model"]) == {"linear"}
