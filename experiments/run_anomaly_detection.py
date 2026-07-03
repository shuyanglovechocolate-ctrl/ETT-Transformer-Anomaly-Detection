"""End-to-end synthetic anomaly detection experiment (Module 6.5).

Ties together Modules 6.1-6.4 over a fixed matrix:

    validation residual -> threshold
    test residual -> inject anomalies -> detect -> point-wise metrics

No model is re-trained; residuals come from Module 6.1 outputs. Produces a
144-row results table plus a few representative figures.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    aggregate_residuals, inject_synthetic_anomalies, compute_threshold,
    detect_anomalies, calculate_detection_metrics, load_prediction_file,
)
from src.anomaly.plots import plot_anomaly_detection

DATASETS = ["ETTh1", "ETTh2"]
HORIZONS = [24, 96]
INPUT_TYPE = "multivariate"
SEED = 42
AGGREGATIONS = ["first", "mean", "max"]
ANOMALY_TYPES = ["spike", "level_shift", "frozen"]
THRESHOLD_METHODS = ["percentile", "mean_std", "iqr", "mad"]

ANOMALY_RATIO = 0.02
MAGNITUDE_SCALE = 3.0
DURATION_RANGES = {"spike": (1, 3), "level_shift": (12, 24), "frozen": (12, 24)}
THRESHOLD_PARAMS = {
    "percentile": {"percentile": 99},
    "mean_std": {"k": 3.0},
    "iqr": {"k": 1.5},
    "mad": {"k": 3.5},
}

RESULT_COLUMNS = [
    "dataset", "model", "input_type", "horizon", "seed", "aggregation_method",
    "anomaly_type", "anomaly_ratio", "magnitude_scale", "threshold_method",
    "threshold_value", "tp", "fp", "tn", "fn", "precision", "recall", "f1",
    "false_positive_rate", "true_negative_rate", "num_points",
    "num_true_anomaly", "num_predicted_anomaly",
    "validation_score_mean", "validation_score_std",
    "test_score_mean", "test_score_std",
]


def best_model_for(best_df, dataset, horizon):
    row = best_df[(best_df.dataset == dataset) & (best_df.input_type == INPUT_TYPE)
                  & (best_df.horizon == horizon)]
    if row.empty:
        raise ValueError(f"No best model for {dataset}/h{horizon}.")
    return row.iloc[0]["best_model"]


def experiment_id(dataset, model, horizon):
    return f"{dataset}_{model}_{INPUT_TYPE}_len96_h{horizon}_seed{SEED}"


def evaluate_scenario(val_scores, test_df, anomaly_type, threshold_method,
                      anomaly_ratio=ANOMALY_RATIO, magnitude_scale=MAGNITUDE_SCALE,
                      seed=SEED):
    """Inject -> threshold (from validation) -> detect -> metrics. Pure/testable."""
    injected = inject_synthetic_anomalies(
        test_df, anomaly_type, anomaly_ratio=anomaly_ratio,
        duration_range=DURATION_RANGES[anomaly_type],
        magnitude_scale=magnitude_scale, seed=seed)
    threshold = compute_threshold(val_scores, threshold_method,
                                  **THRESHOLD_PARAMS[threshold_method])
    detected = detect_anomalies(injected, threshold)
    metrics = calculate_detection_metrics(detected)
    metrics["threshold_value"] = threshold
    return detected, metrics


def run(project_root, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    residual_dir = os.path.join(results_dir, "anomaly", "residuals")
    fig_dir = os.path.join(results_dir, "anomaly", "figures")
    best_df = pd.read_csv(os.path.join(
        results_dir, "metrics", "best_model_by_dataset_horizon.csv"))

    rows = []
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
                    for tmethod in THRESHOLD_METHODS:
                        detected, m = evaluate_scenario(
                            val["anomaly_score"], test, anomaly_type, tmethod)
                        rows.append({
                            "dataset": dataset, "model": model,
                            "input_type": INPUT_TYPE, "horizon": horizon, "seed": SEED,
                            "aggregation_method": agg, "anomaly_type": anomaly_type,
                            "anomaly_ratio": ANOMALY_RATIO,
                            "magnitude_scale": MAGNITUDE_SCALE,
                            "threshold_method": tmethod, **m,
                            "validation_score_mean": float(val["anomaly_score"].mean()),
                            "validation_score_std": float(val["anomaly_score"].std()),
                            "test_score_mean": float(test["anomaly_score"].mean()),
                            "test_score_std": float(test["anomaly_score"].std()),
                        })
                        # Representative figures: ETTh1 / h24 / first / percentile.
                        if (dataset == "ETTh1" and horizon == 24 and agg == "first"
                                and tmethod == "percentile"):
                            fig = os.path.join(
                                fig_dir, f"ETTh1_h24_first_{anomaly_type}_percentile99.png")
                            plot_anomaly_detection(
                                detected, fig,
                                title=f"ETTh1 {model} h24 first — {anomaly_type} (percentile99)")

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    out = os.path.join(results_dir, "anomaly", "metrics",
                       "anomaly_detection_results.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Wrote {out} ({len(results)} rows)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection experiment.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
