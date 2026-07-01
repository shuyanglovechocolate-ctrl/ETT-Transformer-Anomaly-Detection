"""Tests for the result summarization script (Module 3.4b)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.summarize_results import (
    load_experiment_log,
    model_comparison,
    horizon_comparison,
    input_type_comparison,
    summarize,
)


def _write_log(tmp_path, rows) -> str:
    path = tmp_path / "experiment_log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _row(eid, ts, dataset, model, input_type, horizon, mae, rmse, wape):
    return {
        "timestamp": ts, "experiment_id": eid, "dataset": dataset,
        "model": model, "input_type": input_type, "horizon": horizon,
        "mae": mae, "rmse": rmse, "wape": wape,
    }


def test_load_experiment_log_deduplicates_keep_latest(tmp_path):
    # Same experiment_id twice; the later timestamp must win.
    rows = [
        _row("e1", "2026-01-01T00:00:00", "ETTh1", "linear", "multivariate", 24, 2.0, 3.0, 20.0),
        _row("e1", "2026-01-02T00:00:00", "ETTh1", "linear", "multivariate", 24, 1.0, 2.0, 10.0),
    ]
    df = load_experiment_log(_write_log(tmp_path, rows))
    assert len(df) == 1
    assert df.iloc[0]["mae"] == 1.0  # kept the latest


def test_missing_log_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_log(str(tmp_path / "nope.csv"))


def test_model_and_horizon_comparison(tmp_path):
    rows = [
        _row("a1", "t1", "ETTh1", "linear", "multivariate", 24, 1.0, 2.0, 10.0),
        _row("a2", "t2", "ETTh1", "linear", "multivariate", 24, 3.0, 4.0, 30.0),
    ]
    df = load_experiment_log(_write_log(tmp_path, rows))
    mc = model_comparison(df)
    assert mc.iloc[0]["mean_mae"] == 2.0
    assert mc.iloc[0]["num_runs"] == 2
    hc = horizon_comparison(df)
    assert set(["model", "horizon", "mean_mae", "num_runs"]).issubset(hc.columns)


def test_input_type_comparison_improvement(tmp_path):
    # univariate mae 2.0 vs multivariate mae 1.0 -> improvement 1.0, pct 50%.
    rows = [
        _row("u", "t1", "ETTh1", "linear", "univariate", 24, 2.0, 4.0, 20.0),
        _row("m", "t2", "ETTh1", "linear", "multivariate", 24, 1.0, 2.0, 10.0),
    ]
    df = load_experiment_log(_write_log(tmp_path, rows))
    it = input_type_comparison(df)
    assert len(it) == 1
    r = it.iloc[0]
    assert r["univariate_mae"] == 2.0
    assert r["multivariate_mae"] == 1.0
    assert r["mae_improvement"] == 1.0
    assert r["mae_improvement_pct"] == 50.0


def test_summarize_writes_three_tables(tmp_path):
    rows = [
        _row("u", "t1", "ETTh1", "linear", "univariate", 24, 2.0, 4.0, 20.0),
        _row("m", "t2", "ETTh1", "linear", "multivariate", 24, 1.0, 2.0, 10.0),
    ]
    log = _write_log(tmp_path, rows)
    out = tmp_path / "out"
    summarize(log, str(out))
    for name in ("model_comparison", "horizon_comparison", "input_type_comparison"):
        assert (out / f"{name}.csv").exists()
