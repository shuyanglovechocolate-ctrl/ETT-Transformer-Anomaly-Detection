"""Tests for efficiency and complexity analysis (Module 7.6)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.analyze_efficiency_complexity import (
    checkpoint_size_mb,
    aggregate_efficiency,
    model_frontier,
    SUMMARY_COLS,
)


def _run(model, seed, params, mae, ckpt="", input_type="multivariate",
         dataset="ETTh1", horizon=24):
    return {
        "dataset": dataset, "model": model, "input_type": input_type,
        "horizon": horizon, "seed": seed,
        "total_parameters": params, "trainable_parameters": params,
        "mae": mae, "rmse": mae * 1.3, "wape": mae * 10, "epochs_ran": 20,
        "checkpoint_path": ckpt,
    }


def _log(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# checkpoint_size_mb
# ---------------------------------------------------------------------------

def test_checkpoint_size_missing_is_nan():
    assert np.isnan(checkpoint_size_mb(""))
    assert np.isnan(checkpoint_size_mb(np.nan))
    assert np.isnan(checkpoint_size_mb("results/checkpoints/does_not_exist.pt"))


def test_checkpoint_size_reads_real_file(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
    assert checkpoint_size_mb(str(f)) == 2.0


# ---------------------------------------------------------------------------
# aggregate_efficiency
# ---------------------------------------------------------------------------

def test_aggregate_columns_and_grouping():
    rows = [
        _run("dlinear", 42, 10000, 1.40),
        _run("dlinear", 2024, 10000, 1.44),
        _run("naive", 42, 0, 1.50),
    ]
    out = aggregate_efficiency(_log(rows))
    assert list(out.columns) == SUMMARY_COLS
    # dlinear seeds collapse into one row with num_runs == 2
    d = out[out["model"] == "dlinear"].iloc[0]
    assert d["num_runs"] == 2
    assert d["mean_mae"] == 1.42
    assert d["mae_per_1k_params"] == 1.42 / (10000 / 1000.0)


def test_naive_has_nan_per_parameter_metric():
    out = aggregate_efficiency(_log([_run("naive", 42, 0, 1.50)]))
    naive = out[out["model"] == "naive"].iloc[0]
    assert naive["total_parameters"] == 0
    assert np.isnan(naive["mae_per_1k_params"])
    assert np.isnan(naive["checkpoint_size_mb"])  # no checkpoint path


def test_regularized_excluded_by_default_included_on_flag():
    rows = [
        _run("transformer", 42, 50000, 1.55),
        _run("transformer_regularized", 42, 50000, 1.52),
    ]
    default = aggregate_efficiency(_log(rows))
    assert set(default["model"]) == {"transformer"}
    kept = aggregate_efficiency(_log(rows), exclude_substrings=())
    assert set(kept["model"]) == {"transformer", "transformer_regularized"}


def test_missing_checkpoint_path_yields_nan_size():
    rows = [_run("lstm", 42, 30000, 1.60, ckpt="results/checkpoints/nope.pt"),
            _run("lstm", 2024, 30000, 1.58, ckpt="results/checkpoints/nope2.pt")]
    out = aggregate_efficiency(_log(rows))
    assert np.isnan(out.iloc[0]["checkpoint_size_mb"])


def test_model_frontier_aggregates_one_point_per_model():
    summary = aggregate_efficiency(_log([
        _run("dlinear", 42, 10000, 1.40, horizon=24),
        _run("dlinear", 42, 20000, 1.80, horizon=96),
        _run("transformer", 42, 50000, 1.55, horizon=24),
        _run("naive", 42, 0, 1.50),  # excluded: zero params
    ]))
    fr = model_frontier(summary, "total_parameters")
    assert set(fr["model"]) == {"dlinear", "transformer"}  # naive dropped (x==0)
    d = fr[fr["model"] == "dlinear"].iloc[0]
    assert d["x"] == 15000.0            # mean of 10000 and 20000
    assert d["mae"] == 1.60             # mean of 1.40 and 1.80
    assert fr["x"].is_monotonic_increasing  # sorted by x


def test_stray_header_row_is_dropped():
    rows = [_run("dlinear", 42, 10000, 1.40)]
    df = _log(rows)
    # simulate an accidental appended header line
    header_like = {c: c for c in df.columns}
    df = pd.concat([df, pd.DataFrame([header_like])], ignore_index=True)
    out = aggregate_efficiency(df)
    assert set(out["model"]) == {"dlinear"}
