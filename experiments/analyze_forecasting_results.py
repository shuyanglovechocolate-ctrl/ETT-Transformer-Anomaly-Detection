"""Forecasting result analysis (Module 3.5).

Pure post-processing of existing experiment outputs (no model re-runs). Produces:
- per_horizon_metrics.csv    (MAE/RMSE/WAPE per horizon step, per experiment)
- per_horizon_summary.csv    (aggregated across seeds)
- best_model_by_dataset_horizon.csv
- model_significance_tests.csv (paired bootstrap CI on per-window MAE difference)

Reads results/predictions/*.csv and results/metrics/experiment_log.csv.
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_retry(path: str, retries: int = 3, delay: float = 1.0) -> pd.DataFrame:
    """Read a CSV, retrying on transient filesystem errors.

    The sandbox filesystem occasionally raises OSError/TimeoutError or an empty
    read on large files; a short retry makes bulk reads over ~200 files robust.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.experiment import build_experiment_id  # noqa: E402

BASE_MODELS = ["naive", "linear", "nlinear", "dlinear", "lstm", "transformer"]


# ---------------------------------------------------------------------------
# Per-horizon-step metrics
# ---------------------------------------------------------------------------

def per_horizon_metrics_for_experiment(pred_df: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE/WAPE per horizon_index for one prediction dataframe."""
    eps = 1e-6
    rows = []
    for h_idx, g in pred_df.groupby("horizon_index"):
        abs_err = g["abs_residual"].to_numpy()
        resid = g["residual"].to_numpy()
        denom = np.sum(np.abs(g["y_true"].to_numpy())) + eps
        rows.append({
            "horizon_index": int(h_idx),
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "wape": float(np.sum(abs_err) / denom * 100.0),
        })
    return pd.DataFrame(rows)


def compute_per_horizon_metrics(log_df: pd.DataFrame, predictions_dir: str) -> pd.DataFrame:
    """Per-horizon metrics for every experiment that has a predictions file."""
    frames = []
    for _, row in log_df.iterrows():
        pred_path = os.path.join(predictions_dir, f"{row['experiment_id']}_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        pred_df = read_csv_retry(pred_path)
        ph = per_horizon_metrics_for_experiment(pred_df)
        ph.insert(0, "seed", row["seed"])
        ph.insert(0, "horizon", row["horizon"])
        ph.insert(0, "input_type", row["input_type"])
        ph.insert(0, "model", row["model"])
        ph.insert(0, "dataset", row["dataset"])
        frames.append(ph)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_per_horizon(per_horizon_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-horizon metrics across seeds."""
    grouped = per_horizon_df.groupby(
        ["dataset", "model", "input_type", "horizon", "horizon_index"]
    )
    return grouped.agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
        mean_wape=("wape", "mean"), std_wape=("wape", "std"),
        num_runs=("mae", "count"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Best model per setting
# ---------------------------------------------------------------------------

def best_model_by_dataset_horizon(log_df: pd.DataFrame,
                                  exclude_substrings=("regularized",)) -> pd.DataFrame:
    """Pick the lowest-mean-MAE model per (dataset, input_type, horizon)."""
    df = log_df.copy()
    for sub in exclude_substrings:
        df = df[~df["model"].str.contains(sub, na=False)]

    grouped = df.groupby(["dataset", "input_type", "horizon", "model"]).agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
        mean_wape=("wape", "mean"), std_wape=("wape", "std"),
        num_runs=("mae", "count"),
    ).reset_index()

    best_idx = grouped.groupby(["dataset", "input_type", "horizon"])["mean_mae"].idxmin()
    best = grouped.loc[best_idx].rename(columns={"model": "best_model"})
    return best.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paired bootstrap significance on per-window errors
# ---------------------------------------------------------------------------

def bootstrap_mae_diff_ci(errors_a: np.ndarray, errors_b: np.ndarray,
                          n_boot: int = 1000, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """Paired bootstrap CI for MAE(a) - MAE(b) over per-window absolute errors.

    A negative mean_diff means model_a has lower error (is better). The result
    is 'significant' when the CI does not cross zero.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    diff = errors_a - errors_b
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean_diff = float(diff.mean())
    return {
        "mean_abs_error_a": float(errors_a.mean()),
        "mean_abs_error_b": float(errors_b.mean()),
        "mae_diff_a_minus_b": mean_diff,
        "bootstrap_ci_low": float(lo),
        "bootstrap_ci_high": float(hi),
        "a_better": bool(mean_diff < 0),
        "significant": bool(lo > 0 or hi < 0),
        "num_points": int(n),
    }


def _seed_averaged_errors(log_df, predictions_dir, dataset, input_type, horizon, model):
    """Per-position abs error, averaged over seeds, indexed by (sample, h_idx)."""
    rows = log_df[(log_df.dataset == dataset) & (log_df.input_type == input_type)
                  & (log_df.horizon == horizon) & (log_df.model == model)]
    series = []
    for _, r in rows.iterrows():
        pred_path = os.path.join(predictions_dir, f"{r['experiment_id']}_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        df = read_csv_retry(pred_path)
        series.append(df.set_index(["sample_index", "horizon_index"])["abs_residual"])
    if not series:
        return None
    return pd.concat(series, axis=1).mean(axis=1)


def compute_significance_tests(log_df, predictions_dir, input_type="multivariate",
                               models=None, n_boot=1000) -> pd.DataFrame:
    """Pairwise paired-bootstrap comparisons of base models per setting."""
    models = models or BASE_MODELS
    results = []
    settings = log_df[log_df.input_type == input_type][["dataset", "horizon"]]
    settings = settings.drop_duplicates().itertuples(index=False)
    for dataset, horizon in settings:
        err = {}
        for m in models:
            e = _seed_averaged_errors(log_df, predictions_dir, dataset, input_type, horizon, m)
            if e is not None:
                err[m] = e
        for a, b in itertools.combinations([m for m in models if m in err], 2):
            ea, eb = err[a].align(err[b], join="inner")
            res = bootstrap_mae_diff_ci(ea.to_numpy(), eb.to_numpy(), n_boot=n_boot)
            res.update({"dataset": dataset, "input_type": input_type,
                        "horizon": horizon, "model_a": a, "model_b": b})
            results.append(res)
    cols = ["dataset", "input_type", "horizon", "model_a", "model_b",
            "mean_abs_error_a", "mean_abs_error_b", "mae_diff_a_minus_b",
            "bootstrap_ci_low", "bootstrap_ci_high", "a_better", "significant",
            "num_points"]
    return pd.DataFrame(results, columns=cols)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_log(log_path: str) -> pd.DataFrame:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Experiment log not found: {log_path}")
    df = pd.read_csv(log_path)
    return df.sort_values("timestamp").drop_duplicates("experiment_id", keep="last")


def main():
    parser = argparse.ArgumentParser(description="Analyze forecasting results.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    metrics_dir = os.path.join(args.results_dir, "metrics")
    predictions_dir = os.path.join(args.results_dir, "predictions")
    log_df = load_log(os.path.join(metrics_dir, "experiment_log.csv"))

    per_h = compute_per_horizon_metrics(log_df, predictions_dir)
    per_h.to_csv(os.path.join(metrics_dir, "per_horizon_metrics.csv"), index=False)
    summarize_per_horizon(per_h).to_csv(
        os.path.join(metrics_dir, "per_horizon_summary.csv"), index=False)

    best_model_by_dataset_horizon(log_df).to_csv(
        os.path.join(metrics_dir, "best_model_by_dataset_horizon.csv"), index=False)

    sig = compute_significance_tests(log_df, predictions_dir, n_boot=args.n_boot)
    sig.to_csv(os.path.join(metrics_dir, "model_significance_tests.csv"), index=False)

    print("Wrote per_horizon_metrics.csv, per_horizon_summary.csv, "
          "best_model_by_dataset_horizon.csv, model_significance_tests.csv")


if __name__ == "__main__":
    main()
