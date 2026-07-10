"""Forecasting accuracy vs anomaly detection analysis (Module 6.11).

Connects Module 3 and Module 4: does lower forecasting error imply better
residual-based anomaly detection? For each of several models (same scenario), it
runs the residual detector on injected anomalies and joins the detection metrics
with the model's forecasting MAE/RMSE/WAPE, then correlates them.

Guardrails: no re-training; thresholds from validation only; the same injection
positions are shared across models (y_true is model-independent); oracle best-F1
is an upper bound; residual detector only (no hybrid, to avoid confounds).
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    inject_synthetic_anomalies, load_prediction_file, compute_threshold,
    detect_anomalies, calculate_detection_metrics,
    calculate_event_detection_metrics, pr_auc, best_f1_threshold,
)
from experiments.prepare_multimodel_residuals import (
    MODELS, DATASETS, HORIZONS, AGGREGATION, experiment_id,
    build_multimodel_scenarios,
)
from experiments.run_anomaly_detection import DURATION_RANGES, ANOMALY_RATIO

ANOMALY_TYPES = ["spike", "level_shift", "frozen"]
INJECTION_SEEDS = [42, 2024, 3407]
PERCENTILE = 99

FORECAST_METRICS = ["forecast_mae", "forecast_rmse", "forecast_wape"]
DETECTION_METRICS = ["average_precision", "oracle_best_f1", "fixed_f1", "event_recall"]


def load_forecast_metrics(log_path):
    """Map experiment_id -> {mae, rmse, wape} from the seed-42 experiment log."""
    log = pd.read_csv(log_path).sort_values("timestamp").drop_duplicates(
        "experiment_id", keep="last")
    return {r["experiment_id"]: {"forecast_mae": r["mae"], "forecast_rmse": r["rmse"],
                                 "forecast_wape": r["wape"]}
            for _, r in log.iterrows()}


def detection_metrics_for(val_df, injected):
    """Residual detector metrics on one injected series."""
    val_scores = val_df["anomaly_score"].to_numpy()
    test_scores = injected["anomaly_score"].to_numpy()
    labels = injected["is_anomaly"].to_numpy()

    threshold = compute_threshold(val_scores, "percentile", percentile=PERCENTILE)
    detected = detect_anomalies(injected, threshold, score_col="anomaly_score")
    point = calculate_detection_metrics(detected)
    event = calculate_event_detection_metrics(detected)
    return {
        "average_precision": pr_auc(labels, test_scores),
        "oracle_best_f1": best_f1_threshold(labels, test_scores)["best_f1"],
        "fixed_f1": point["f1"],
        "fixed_precision": point["precision"],
        "fixed_recall": point["recall"],
        "event_recall": event["event_recall"],
        "mean_detection_delay": event["mean_detection_delay"],
    }


def compute_correlation_table(df):
    """Per (dataset, horizon, anomaly_type): correlate forecast vs detection metrics
    across models (averaged over injection seeds)."""
    rows = []
    grouped = df.groupby(["dataset", "horizon", "anomaly_type", "model"]).mean(
        numeric_only=True).reset_index()
    for (dataset, horizon, atype), g in grouped.groupby(["dataset", "horizon", "anomaly_type"]):
        if len(g) < 3:
            continue
        for fcol in FORECAST_METRICS:
            for dcol in DETECTION_METRICS:
                x = g[fcol].to_numpy()
                y = g[dcol].to_numpy()
                if np.std(x) == 0 or np.std(y) == 0:
                    pear = spear = float("nan")
                else:
                    pear = float(pearsonr(x, y)[0])
                    spear = float(spearmanr(x, y)[0])
                rows.append({"dataset": dataset, "horizon": horizon,
                             "anomaly_type": atype, "forecast_metric": fcol,
                             "detection_metric": dcol, "pearson_correlation": pear,
                             "spearman_correlation": spear, "num_models": len(g)})
    return pd.DataFrame(rows)


def run(project_root, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    residual_dir = os.path.join(results_dir, "anomaly", "multimodel_residuals")
    metrics_dir = os.path.join(results_dir, "anomaly", "metrics")
    fig_dir = os.path.join(results_dir, "anomaly", "figures")

    forecast = load_forecast_metrics(
        os.path.join(results_dir, "metrics", "experiment_log.csv"))

    rows = []
    for dataset, model_name, horizon in build_multimodel_scenarios():
        eid = experiment_id(dataset, model_name, horizon)
        val = load_prediction_file(os.path.join(residual_dir, f"{eid}_val_{AGGREGATION}.csv"))
        test = load_prediction_file(os.path.join(residual_dir, f"{eid}_test_{AGGREGATION}.csv"))
        fc = forecast.get(eid, {})
        for anomaly_type in ANOMALY_TYPES:
            for inj_seed in INJECTION_SEEDS:
                injected = inject_synthetic_anomalies(
                    test, anomaly_type, anomaly_ratio=ANOMALY_RATIO,
                    duration_range=DURATION_RANGES[anomaly_type], seed=inj_seed)
                dm = detection_metrics_for(val, injected)
                rows.append({"dataset": dataset, "horizon": horizon, "model": model_name,
                             "anomaly_type": anomaly_type, "injection_seed": inj_seed,
                             **fc, **dm})

    results = pd.DataFrame(rows)
    os.makedirs(metrics_dir, exist_ok=True)
    results.to_csv(os.path.join(metrics_dir, "accuracy_vs_detection.csv"), index=False)

    corr = compute_correlation_table(results)
    corr.to_csv(os.path.join(metrics_dir, "accuracy_detection_correlation.csv"), index=False)
    print(f"Wrote accuracy_vs_detection.csv ({len(results)} rows) and correlation table.")

    _plots(results, fig_dir)


def _plots(results, fig_dir):
    from src.viz import apply_paper_style, color_for_model
    apply_paper_style()
    os.makedirs(fig_dir, exist_ok=True)
    sub = results[(results.dataset == "ETTh1") & (results.horizon == 24)]
    agg = sub.groupby("model").mean(numeric_only=True).reindex(MODELS)
    for dcol, fname in [("average_precision", "accuracy_vs_detection_pr_auc.png"),
                        ("fixed_f1", "accuracy_vs_detection_f1.png")]:
        plt.figure(figsize=(8, 6))
        for model in MODELS:
            if model in agg.index:
                plt.scatter(agg.loc[model, "forecast_mae"], agg.loc[model, dcol],
                            s=90, label=model, color=color_for_model(model),
                            edgecolor="white", linewidth=0.8, zorder=3)
        plt.title(f"Forecasting MAE vs detection {dcol} (ETTh1 h24, mean over anomaly types/seeds)")
        plt.xlabel("forecasting MAE (lower = better forecast)")
        plt.ylabel(dcol)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, fname))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Accuracy vs detection analysis.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
