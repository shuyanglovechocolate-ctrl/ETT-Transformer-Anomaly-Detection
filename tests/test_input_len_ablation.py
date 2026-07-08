"""Tests for the input-length ablation analysis (Module 8.3)."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.analyze_input_len_ablation import (
    build_ablation_table, ranking_by_input_len, is_ranking_stable,
    build_summary, TABLE_COLS,
)


def _row(model, input_len, mae, params=50000):
    return {
        "dataset": "ETTh1", "model": model, "input_type": "multivariate",
        "input_len": input_len, "horizon": 96, "seed": 42,
        "mae": mae, "rmse": mae * 1.3, "wape": mae * 10, "total_parameters": params,
    }


def _log(rows):
    return pd.DataFrame(rows)


def test_table_columns_and_rank():
    df = _log([
        _row("dlinear", 96, 1.30), _row("nlinear", 96, 1.40),
        _row("transformer", 96, 1.55),
    ])
    out = build_ablation_table(df)
    assert list(out.columns) == TABLE_COLS
    r = out[out["input_len"] == 96].set_index("model")["rank_within_input_len"]
    assert r["dlinear"] == 1 and r["nlinear"] == 2 and r["transformer"] == 3


def test_ranking_stable_when_order_matches():
    rows = []
    for il in (48, 96, 192):
        rows += [_row("dlinear", il, 1.30), _row("nlinear", il, 1.40),
                 _row("transformer", il, 1.55)]
    table = build_ablation_table(_log(rows))
    assert is_ranking_stable(table) is True
    summary = build_summary(table)
    assert bool(summary["ranking_stable_across_input_len"].all()) is True
    assert set(summary["best_model"]) == {"dlinear"}


def test_ranking_unstable_when_order_flips():
    rows = [
        _row("dlinear", 48, 1.30), _row("transformer", 48, 1.40),  # dlinear best
        _row("dlinear", 192, 1.50), _row("transformer", 192, 1.40),  # transformer best
    ]
    table = build_ablation_table(_log(rows))
    assert is_ranking_stable(table) is False
    ranks = ranking_by_input_len(table)
    assert ranks[48][0] == "dlinear"
    assert ranks[192][0] == "transformer"
