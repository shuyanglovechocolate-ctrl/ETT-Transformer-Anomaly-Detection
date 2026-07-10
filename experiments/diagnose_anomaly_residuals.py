"""Residual and frozen-flatness diagnostics (Module 6.8).

Diagnostic-only (no new detector, no matrix expansion). Explains:
- the shape of residual anomaly-score distributions (skew, autocorrelation),
  which underlies the false-positive behaviour of simple thresholds;
- why frozen anomalies are weakly separated by residual magnitude but clearly
  separated by a causal temporal-flatness feature.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    load_prediction_file, inject_synthetic_anomalies,
    summarize_score_distribution, compute_flatness_score, compare_scores_by_label,
)
from src.viz import PALETTE, apply_paper_style

DATASETS = ["ETTh1", "ETTh2"]
HORIZONS = [24, 96]
INPUT_TYPE = "multivariate"
MODEL_SEED = 42
AGGREGATIONS = ["first", "mean", "max"]
FLATNESS_WINDOW = 12
FROZEN_DURATION = (12, 24)
ANOMALY_RATIO = 0.02
INJECTION_SEED = 42


def best_model_for(best_df, dataset, horizon):
    row = best_df[(best_df.dataset == dataset) & (best_df.input_type == INPUT_TYPE)
                  & (best_df.horizon == horizon)]
    return row.iloc[0]["best_model"]


def experiment_id(dataset, model, horizon):
    return f"{dataset}_{model}_{INPUT_TYPE}_len96_h{horizon}_seed{MODEL_SEED}"


def run(project_root, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    residual_dir = os.path.join(results_dir, "anomaly", "residuals")
    metrics_dir = os.path.join(results_dir, "anomaly", "metrics")
    fig_dir = os.path.join(results_dir, "anomaly", "figures")
    best_df = pd.read_csv(os.path.join(results_dir, "metrics",
                                       "best_model_by_dataset_horizon.csv"))

    residual_rows, flatness_rows = [], []
    fig_source = None

    for dataset in DATASETS:
        for horizon in HORIZONS:
            model = best_model_for(best_df, dataset, horizon)
            eid = experiment_id(dataset, model, horizon)
            for agg in AGGREGATIONS:
                for split in ("validation", "test"):
                    tag = "val" if split == "validation" else "test"
                    df = load_prediction_file(
                        os.path.join(residual_dir, f"{eid}_{tag}_residual_{agg}.csv"))
                    stats = summarize_score_distribution(df, "anomaly_score")
                    residual_rows.append({
                        "dataset": dataset, "model": model, "horizon": horizon,
                        "aggregation_method": agg, "split": split, **stats})

                # Frozen flatness diagnostic on the test series.
                test = load_prediction_file(
                    os.path.join(residual_dir, f"{eid}_test_residual_{agg}.csv"))
                injected = inject_synthetic_anomalies(
                    test, "frozen", anomaly_ratio=ANOMALY_RATIO,
                    duration_range=FROZEN_DURATION, seed=INJECTION_SEED)
                injected = injected.assign(
                    flatness_score=compute_flatness_score(
                        injected["y_true_anomalous"], window=FLATNESS_WINDOW).to_numpy())
                valid = injected.dropna(subset=["flatness_score"])
                cmp = compare_scores_by_label(
                    valid, ["anomaly_score", "flatness_score"], "is_anomaly")
                res = cmp[cmp.score_col == "anomaly_score"].iloc[0]
                flat = cmp[cmp.score_col == "flatness_score"].iloc[0]
                flatness_rows.append({
                    "dataset": dataset, "horizon": horizon, "aggregation_method": agg,
                    "anomaly_type": "frozen", "window": FLATNESS_WINDOW,
                    "mean_flatness_score_anomaly": flat["mean_anomaly"],
                    "mean_flatness_score_normal": flat["mean_normal"],
                    "ratio_anomaly_to_normal": flat["ratio_anomaly_to_normal"],
                    "mean_residual_score_anomaly": res["mean_anomaly"],
                    "mean_residual_score_normal": res["mean_normal"],
                    "residual_ratio_anomaly_to_normal": res["ratio_anomaly_to_normal"],
                })
                if dataset == "ETTh1" and horizon == 24 and agg == "first":
                    fig_source = injected

    pd.DataFrame(residual_rows).to_csv(
        os.path.join(metrics_dir, "residual_diagnostics.csv"), index=False)
    pd.DataFrame(flatness_rows).to_csv(
        os.path.join(metrics_dir, "frozen_flatness_diagnostics.csv"), index=False)
    print("Wrote residual_diagnostics.csv, frozen_flatness_diagnostics.csv")

    _plot_distribution(residual_dir, fig_dir, best_df)
    if fig_source is not None:
        _plot_frozen(fig_source, fig_dir)


def _plot_distribution(residual_dir, fig_dir, best_df):
    """Histogram of validation vs test residual anomaly scores (ETTh1 h24 first)."""
    os.makedirs(fig_dir, exist_ok=True)
    apply_paper_style()
    model = best_model_for(best_df, "ETTh1", 24)
    eid = experiment_id("ETTh1", model, 24)
    val = load_prediction_file(os.path.join(residual_dir, f"{eid}_val_residual_first.csv"))
    test = load_prediction_file(os.path.join(residual_dir, f"{eid}_test_residual_first.csv"))
    plt.figure(figsize=(8, 5))
    plt.hist(val["anomaly_score"], bins=60, alpha=0.6, label="validation", density=True)
    plt.hist(test["anomaly_score"], bins=60, alpha=0.6, label="test (clean)", density=True)
    plt.yscale("log")
    plt.title("Residual anomaly-score distribution (ETTh1 h24 first)")
    plt.xlabel("anomaly score (|residual|)")
    plt.ylabel("density (log)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "residual_score_distribution.png"))
    plt.close()


def _plot_frozen(injected, fig_dir):
    """Residual score vs flatness score over time for a frozen scenario."""
    os.makedirs(fig_dir, exist_ok=True)
    apply_paper_style()
    dates = pd.to_datetime(injected["target_date"])
    is_anom = injected["is_anomaly"].to_numpy(dtype=bool)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(dates, injected["anomaly_score"], color=PALETTE[0])
    ax1.scatter(dates[is_anom], injected["anomaly_score"].to_numpy()[is_anom],
                color=PALETTE[1], s=16, zorder=5, label="frozen (true)")
    ax1.set_ylabel("residual score")
    ax1.set_title("Frozen anomaly: residual score is weak, flatness score is clear")
    ax1.legend(loc="upper right")
    ax2.plot(dates, injected["flatness_score"], color=PALETTE[2])
    ax2.scatter(dates[is_anom], injected["flatness_score"].to_numpy()[is_anom],
                color=PALETTE[1], s=16, zorder=5, label="frozen (true)")
    ax2.set_yscale("log")
    ax2.set_ylabel("flatness score (log)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "frozen_flatness_diagnostic.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Anomaly residual diagnostics.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
