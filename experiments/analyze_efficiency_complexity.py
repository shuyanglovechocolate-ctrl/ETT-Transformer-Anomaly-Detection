"""Efficiency and complexity analysis (Module 7.6).

Pure post-hoc analysis of existing forecasting runs (no retraining, no model
changes). Reuses the parameter counts and errors already recorded in
`results/metrics/experiment_log.csv` and the on-disk checkpoint sizes to quantify
the accuracy-vs-complexity trade-off across models.

The goal is to strengthen the forecasting conclusion: linear-family models are not
only more accurate under the tested protocol, but also substantially more
parameter-efficient and cheaper to store than the recurrent and Transformer
baselines.

Reads  results/metrics/experiment_log.csv  (+ results/checkpoints/*.pt sizes)
Writes results/metrics/efficiency_complexity_summary.csv
       results/figures/efficiency_mae_vs_params.png
       results/figures/efficiency_mae_vs_checkpoint_size.png

The zero-parameter Naive baseline is included in the table for completeness but is
excluded from per-parameter efficiency metrics and the complexity figures (a
persistence rule has no trainable parameters or checkpoint).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

GROUP_KEY = ["dataset", "model", "input_type", "horizon"]
SUMMARY_COLS = [
    "dataset", "model", "horizon", "input_type", "num_runs",
    "mean_mae", "std_mae", "mean_rmse", "mean_wape",
    "total_parameters", "trainable_parameters", "mae_per_1k_params",
    "checkpoint_size_mb", "mean_epochs_ran",
]


def read_csv_retry(path: str, retries: int = 3, delay: float = 1.0) -> pd.DataFrame:
    """Read a CSV, retrying on transient sandbox filesystem errors."""
    last_exc = None
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc


def checkpoint_size_mb(path) -> float:
    """Size of a checkpoint file in MB, or NaN if it is missing / not recorded."""
    if not isinstance(path, str) or not path:
        return float("nan")
    if not os.path.exists(path):
        return float("nan")
    return os.path.getsize(path) / (1024.0 * 1024.0)


def aggregate_efficiency(log_df: pd.DataFrame,
                         exclude_substrings=("regularized",)) -> pd.DataFrame:
    """Aggregate per-run metrics into an accuracy-vs-complexity summary.

    One row per (dataset, model, input_type, horizon), averaged across seeds.
    `mae_per_1k_params` and `checkpoint_size_mb` are NaN for zero-parameter or
    checkpoint-less models (the Naive baseline).
    """
    df = log_df.copy()
    # Defensive: drop any accidental repeated-header row appended to the log.
    df = df[df["model"].astype(str) != "model"]
    for sub in exclude_substrings:
        df = df[~df["model"].astype(str).str.contains(sub, na=False)]

    for col in ("total_parameters", "trainable_parameters", "mae", "rmse",
                "wape", "epochs_ran"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mae"])
    df["_ckpt_mb"] = df["checkpoint_path"].apply(checkpoint_size_mb)

    rows = []
    for (dataset, model, input_type, horizon), g in df.groupby(GROUP_KEY):
        total_params = int(g["total_parameters"].iloc[0])
        mean_mae = float(g["mae"].mean())
        ckpt = g["_ckpt_mb"]
        rows.append({
            "dataset": dataset,
            "model": model,
            "horizon": int(horizon),
            "input_type": input_type,
            "num_runs": int(len(g)),
            "mean_mae": mean_mae,
            "std_mae": float(g["mae"].std(ddof=1)) if len(g) > 1 else float("nan"),
            "mean_rmse": float(g["rmse"].mean()),
            "mean_wape": float(g["wape"].mean()),
            "total_parameters": total_params,
            "trainable_parameters": int(g["trainable_parameters"].iloc[0]),
            "mae_per_1k_params": (mean_mae / (total_params / 1000.0)
                                  if total_params > 0 else float("nan")),
            "checkpoint_size_mb": (float(ckpt.mean()) if ckpt.notna().any()
                                   else float("nan")),
            "mean_epochs_ran": float(g["epochs_ran"].mean()),
        })
    out = pd.DataFrame(rows, columns=SUMMARY_COLS)
    return out.sort_values(["dataset", "horizon", "input_type", "mean_mae"]) \
        .reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def model_frontier(summary: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """One point per model: mean of `x_col` and mean/std of mean_mae across settings.

    Aggregating to the model level keeps the complexity axis interpretable instead
    of conflating it with per-setting (horizon / dataset / input-type) MAE variance.
    """
    data = summary.dropna(subset=[x_col, "mean_mae"])
    data = data[data[x_col] > 0]
    agg = data.groupby("model").agg(
        x=(x_col, "mean"), mae=("mean_mae", "mean"),
        mae_sd=("mean_mae", "std"), n=("mean_mae", "size"),
    ).reset_index()
    return agg.sort_values("x").reset_index(drop=True)


def _frontier_plot(ax, summary: pd.DataFrame, x_col: str, xlabel: str):
    from src.viz import color_for_model
    agg = model_frontier(summary, x_col)
    prev_x = None
    prev_y = None
    for _, r in agg.iterrows():
        yerr = 0.0 if pd.isna(r["mae_sd"]) else r["mae_sd"]
        color = color_for_model(r["model"])
        ax.errorbar(r["x"], r["mae"], yerr=yerr, fmt="o", ms=9, capsize=4,
                    color=color)
        # Stagger the label when this point sits almost on top of the previous
        # one (e.g. linear vs nlinear share ~the same parameter count) so the
        # two names do not print over each other.
        dy = 8
        if (prev_x is not None
                and abs(r["x"] - prev_x) / max(prev_x, 1e-9) < 0.15
                and abs(r["mae"] - prev_y) < 0.25):
            dy = -14
        ax.annotate(r["model"], (r["x"], r["mae"]), color=color,
                    textcoords="offset points", xytext=(10, dy), fontsize=9,
                    fontweight="bold")
        prev_x, prev_y = r["x"], r["mae"]
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean MAE (original OT scale)")
    ax.grid(True, which="both", alpha=0.3)


def make_figures(summary: pd.DataFrame, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.viz import apply_paper_style

    apply_paper_style()
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    _frontier_plot(ax, summary, "total_parameters", "Trainable parameters (log scale)")
    ax.set_title("Forecasting accuracy vs model complexity\n"
                 "(per-model mean; error bars = MAE std across settings)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "efficiency_mae_vs_params.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _frontier_plot(ax, summary, "checkpoint_size_mb", "Checkpoint size (MB, log scale)")
    ax.set_title("Forecasting accuracy vs checkpoint size\n"
                 "(per-model mean; error bars = MAE std across settings)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "efficiency_mae_vs_checkpoint_size.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Efficiency and complexity analysis (post-hoc).")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--include-regularized", action="store_true",
                        help="Include *_regularized robustness variants.")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    metrics_dir = os.path.join(args.results_dir, "metrics")
    fig_dir = os.path.join(args.results_dir, "figures")
    log_path = os.path.join(metrics_dir, "experiment_log.csv")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Experiment log not found: {log_path}")

    log_df = read_csv_retry(log_path)
    exclude = () if args.include_regularized else ("regularized",)
    summary = aggregate_efficiency(log_df, exclude_substrings=exclude)

    out_path = os.path.join(metrics_dir, "efficiency_complexity_summary.csv")
    summary.to_csv(out_path, index=False)
    msg = f"Wrote {out_path} ({len(summary)} rows)"
    if not args.no_figures:
        make_figures(summary, fig_dir)
        msg += " + efficiency_mae_vs_params.png, efficiency_mae_vs_checkpoint_size.png"
    print(msg)


if __name__ == "__main__":
    main()
