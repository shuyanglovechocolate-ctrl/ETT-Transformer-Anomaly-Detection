"""Threshold-free anomaly detection evaluation (Module 6.9).

Evaluates each detector's continuous anomaly score with PR-AUC / ROC-AUC / best-F1
and F1 at target false-positive rates, removing the dependence on a specific
threshold method. Reuses the shared detector-scoring dispatch and injection chain
(no re-training). Also draws a PR-curve figure for one representative scenario.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    inject_synthetic_anomalies, load_prediction_file, score_detector,
    calculate_threshold_free_metrics,
)
from experiments.run_anomaly_detection import (
    best_model_for, experiment_id, DATASETS, HORIZONS, AGGREGATIONS,
    ANOMALY_TYPES, INJECTION_SEEDS, DETECTORS, DURATION_RANGES, ANOMALY_RATIO,
)

RESULT_COLUMNS = [
    "dataset", "model", "horizon", "aggregation_method", "anomaly_type",
    "injection_seed", "detector_type",
    "pr_auc", "roc_auc", "best_f1", "best_f1_threshold",
    "f1_at_fpr_1pct", "recall_at_fpr_1pct",
    "f1_at_fpr_2pct", "recall_at_fpr_2pct",
    "f1_at_fpr_5pct", "recall_at_fpr_5pct",
]


def run(project_root, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    residual_dir = os.path.join(results_dir, "anomaly", "residuals")
    metrics_dir = os.path.join(results_dir, "anomaly", "metrics")
    fig_dir = os.path.join(results_dir, "anomaly", "figures")
    best_df = pd.read_csv(os.path.join(
        results_dir, "metrics", "best_model_by_dataset_horizon.csv"))

    rows = []
    pr_curve_source = None
    for dataset in DATASETS:
        for horizon in HORIZONS:
            model = best_model_for(best_df, dataset, horizon)
            eid = experiment_id(dataset, model, horizon)
            for agg in AGGREGATIONS:
                val = load_prediction_file(
                    os.path.join(residual_dir, f"{eid}_val_residual_{agg}.csv"))
                test = load_prediction_file(
                    os.path.join(residual_dir, f"{eid}_test_residual_{agg}.csv"))
                for anomaly_type in ANOMALY_TYPES:
                    for inj_seed in INJECTION_SEEDS:
                        injected = inject_synthetic_anomalies(
                            test, anomaly_type, anomaly_ratio=ANOMALY_RATIO,
                            duration_range=DURATION_RANGES[anomaly_type], seed=inj_seed)
                        labels = injected["is_anomaly"].to_numpy()
                        for detector in DETECTORS:
                            _, test_scores = score_detector(detector, injected, val)
                            m = calculate_threshold_free_metrics(labels, test_scores)
                            rows.append({
                                "dataset": dataset, "model": model, "horizon": horizon,
                                "aggregation_method": agg, "anomaly_type": anomaly_type,
                                "injection_seed": inj_seed, "detector_type": detector, **m})
                            if (dataset == "ETTh1" and horizon == 24 and agg == "first"
                                    and anomaly_type == "spike" and inj_seed == 42):
                                pr_curve_source = pr_curve_source or {}
                                pr_curve_source[detector] = (labels, test_scores)

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    os.makedirs(metrics_dir, exist_ok=True)
    results.to_csv(os.path.join(metrics_dir, "anomaly_threshold_free_results.csv"), index=False)

    summary = results.groupby(["detector_type", "anomaly_type"]).agg(
        mean_pr_auc=("pr_auc", "mean"), std_pr_auc=("pr_auc", "std"),
        mean_roc_auc=("roc_auc", "mean"), mean_best_f1=("best_f1", "mean"),
        num_runs=("pr_auc", "count"),
    ).reset_index()
    summary.to_csv(os.path.join(metrics_dir, "anomaly_threshold_free_summary.csv"), index=False)
    print(f"Wrote anomaly_threshold_free_results.csv ({len(results)} rows) and summary.")

    if pr_curve_source:
        _plot_pr_curves(pr_curve_source, fig_dir)


def _plot_pr_curves(sources, fig_dir):
    from src.viz import apply_paper_style
    apply_paper_style()
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for detector, (labels, scores) in sources.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        plt.plot(recall, precision, label=detector)
    plt.title("Precision-Recall curves (ETTh1 h24 first, spike)")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pr_curves_spike.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Threshold-free anomaly evaluation.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
