"""ETTm external-validity analysis (minute-level frequency effect).

Post-processes an ISOLATED ETTm training run (produced by run_matrix.py with
``--results-dir results_ettm``) and writes a small forecasting comparison table to
``results/external_validity/``. It deliberately reads a separate results directory and
writes separate outputs, so the canonical ETTh1/ETTh2 results backing the v1.0-thesis
release are never touched.

The question is narrow: on 15-minute ETTm1 / ETTm2 data, does the core finding — that
linear-family models forecast at least as well as the Transformer — still hold? This is
a lightweight single-seed smoke test of external validity, not a full re-run of the
core study.

Reads  results_ettm/metrics/experiment_log.csv
Writes results/external_validity/ettm_forecasting_comparison.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

LINEAR_FAMILY = ("linear", "nlinear", "dlinear")
OUTPUT_COLS = ["dataset", "model", "horizon", "input_type", "num_runs",
               "mean_mae", "std_mae", "mean_rmse", "mean_wape", "total_parameters",
               "rank_within_dataset"]


def read_csv_retry(path: str, retries: int = 3, delay: float = 1.0) -> pd.DataFrame:
    last_exc = None
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc


def build_comparison(log_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model, input_type, horizon), ranked by MAE per dataset."""
    df = log_df[log_df["model"].astype(str) != "model"].copy()
    for col in ("mae", "rmse", "wape", "total_parameters"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mae"])

    rows = []
    for (dataset, model, input_type, horizon), g in df.groupby(
            ["dataset", "model", "input_type", "horizon"]):
        rows.append({
            "dataset": dataset, "model": model, "horizon": int(horizon),
            "input_type": input_type, "num_runs": int(len(g)),
            "mean_mae": float(g["mae"].mean()),
            "std_mae": float(g["mae"].std(ddof=1)) if len(g) > 1 else 0.0,
            "mean_rmse": float(g["rmse"].mean()),
            "mean_wape": float(g["wape"].mean()),
            "total_parameters": int(g["total_parameters"].iloc[0]),
        })
    out = pd.DataFrame(rows)
    out["rank_within_dataset"] = (out.groupby("dataset")["mean_mae"]
                                  .rank(method="min").astype(int))
    out = out.sort_values(["dataset", "mean_mae"]).reset_index(drop=True)
    return out[OUTPUT_COLS]


def verdict(comparison: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: best linear-family MAE vs Transformer MAE, and who wins."""
    rows = []
    for dataset, g in comparison.groupby("dataset"):
        lin = g[g["model"].isin(LINEAR_FAMILY)]
        trans = g[g["model"] == "transformer"]
        best_lin = float(lin["mean_mae"].min()) if len(lin) else float("nan")
        best_lin_model = (lin.loc[lin["mean_mae"].idxmin(), "model"]
                          if len(lin) else "")
        trans_mae = float(trans["mean_mae"].iloc[0]) if len(trans) else float("nan")
        rows.append({
            "dataset": dataset,
            "best_linear_model": best_lin_model,
            "best_linear_mae": best_lin,
            "transformer_mae": trans_mae,
            "linear_beats_transformer": (bool(best_lin < trans_mae)
                                         if len(lin) and len(trans) else None),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="ETTm external-validity analysis.")
    parser.add_argument("--ettm-results-dir", default=str(PROJECT_ROOT / "results_ettm"))
    parser.add_argument("--out-dir",
                        default=str(PROJECT_ROOT / "results" / "external_validity"))
    args = parser.parse_args()

    log_path = os.path.join(args.ettm_results_dir, "metrics", "experiment_log.csv")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"ETTm experiment log not found: {log_path}")

    log_df = read_csv_retry(log_path)
    comparison = build_comparison(log_df)
    v = verdict(comparison)

    os.makedirs(args.out_dir, exist_ok=True)
    comparison.to_csv(os.path.join(args.out_dir, "ettm_forecasting_comparison.csv"),
                      index=False)
    v.to_csv(os.path.join(args.out_dir, "ettm_forecasting_verdict.csv"), index=False)
    print(f"Wrote ettm_forecasting_comparison.csv ({len(comparison)} rows) and verdict.")
    print(v.to_string(index=False))


if __name__ == "__main__":
    main()
