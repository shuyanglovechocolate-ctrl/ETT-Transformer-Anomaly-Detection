"""Summarize experiment results into comparison tables (Module 3.4b).

Reads results/metrics/experiment_log.csv and writes three tables:
- model_comparison.csv     (mean/std MAE/RMSE/WAPE per dataset/model/input_type/horizon)
- horizon_comparison.csv   (mean/std per model/horizon)
- input_type_comparison.csv (univariate vs multivariate improvement)

The log is append-only, so rows are de-duplicated by experiment_id (keeping the
latest timestamp) before any aggregation, otherwise re-runs would double-count.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRIC_COLS = ["mae", "rmse", "wape"]


def load_experiment_log(log_path: str) -> pd.DataFrame:
    """Load and de-duplicate the experiment log (keep latest per experiment_id)."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Experiment log not found: {log_path}. "
            "Run experiments first (experiments/run_matrix.py)."
        )
    df = pd.read_csv(log_path)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["experiment_id"], keep="last"
    )
    return df


def model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of each metric per dataset, model, input_type and horizon."""
    grouped = df.groupby(["dataset", "model", "input_type", "horizon"])
    out = grouped.agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
        mean_wape=("wape", "mean"), std_wape=("wape", "std"),
        num_runs=("mae", "count"),
    ).reset_index()
    return out


def horizon_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of each metric per model and horizon."""
    grouped = df.groupby(["model", "horizon"])
    out = grouped.agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
        mean_wape=("wape", "mean"), std_wape=("wape", "std"),
        num_runs=("mae", "count"),
    ).reset_index()
    return out


def input_type_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare univariate vs multivariate per dataset, model and horizon."""
    base = df.groupby(["dataset", "model", "horizon", "input_type"]).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), wape=("wape", "mean"),
        n=("mae", "count"),
    ).reset_index()

    uni = base[base["input_type"] == "univariate"]
    mul = base[base["input_type"] == "multivariate"]
    merged = pd.merge(
        uni, mul, on=["dataset", "model", "horizon"],
        suffixes=("_uni", "_mul"), how="inner",
    )

    result = pd.DataFrame({
        "dataset": merged["dataset"],
        "model": merged["model"],
        "horizon": merged["horizon"],
    })
    for metric in METRIC_COLS:
        uni_col = merged[f"{metric}_uni"]
        mul_col = merged[f"{metric}_mul"]
        improvement = uni_col - mul_col
        # Positive improvement means multivariate is better. Guard divide-by-zero.
        pct = np.where(uni_col != 0, improvement / uni_col * 100.0, 0.0)
        result[f"univariate_{metric}"] = uni_col.values
        result[f"multivariate_{metric}"] = mul_col.values
        result[f"{metric}_improvement"] = improvement.values
        result[f"{metric}_improvement_pct"] = pct
    result["num_univariate_runs"] = merged["n_uni"].values
    result["num_multivariate_runs"] = merged["n_mul"].values
    return result


def summarize(log_path: str, output_dir: str) -> dict:
    """Generate and save all three comparison tables. Returns the DataFrames."""
    df = load_experiment_log(log_path)
    os.makedirs(output_dir, exist_ok=True)

    tables = {
        "model_comparison": model_comparison(df),
        "horizon_comparison": horizon_comparison(df),
        "input_type_comparison": input_type_comparison(df),
    }
    for name, table in tables.items():
        path = os.path.join(output_dir, f"{name}.csv")
        table.to_csv(path, index=False)
        print(f"Wrote {path} ({len(table)} rows)")
    return tables


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument(
        "--log-path",
        default=str(PROJECT_ROOT / "results" / "metrics" / "experiment_log.csv"),
        help="Path to experiment_log.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "metrics"),
        help="Directory for the comparison CSVs.",
    )
    args = parser.parse_args()
    summarize(args.log_path, args.output_dir)


if __name__ == "__main__":
    main()
