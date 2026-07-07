"""End-to-end synthetic anomaly detection experiment (Modules 6.5 / 6.6).

Compares the forecast-residual detector against causal statistical baselines
(raw_zscore, diff_score, rolling_zscore) over a fixed matrix, repeated across
several injection seeds for robustness. All detectors share the SAME injected
test set per scenario, and every threshold is learned only from validation
scores (leakage-free). No model is re-trained.

Matrix: 2 datasets x 2 horizons x 3 aggregations x 3 anomaly types
        x 3 injection seeds x 4 detectors x 4 thresholds = 1728 rows.
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
    inject_synthetic_anomalies, compute_threshold, detect_anomalies,
    calculate_detection_metrics, load_prediction_file,
    calculate_event_detection_metrics, score_detector,
)
from src.anomaly.plots import plot_anomaly_detection

DATASETS = ["ETTh1", "ETTh2"]
HORIZONS = [24, 96]
INPUT_TYPE = "multivariate"
MODEL_SEED = 42
AGGREGATIONS = ["first", "mean", "max"]
ANOMALY_TYPES = ["spike", "level_shift", "frozen"]
INJECTION_SEEDS = [42, 2024, 3407]
DETECTORS = ["residual", "raw_zscore", "diff_score", "rolling_zscore"]
THRESHOLD_METHODS = ["percentile", "mean_std", "iqr", "mad"]

ANOMALY_RATIO = 0.02
MAGNITUDE_SCALE = 3.0
ROLLING_WINDOW = 24
DURATION_RANGES = {"spike": (1, 3), "level_shift": (12, 24), "frozen": (12, 24)}
THRESHOLD_PARAMS = {
    "percentile": {"percentile": 99}, "mean_std": {"k": 3.0},
    "iqr": {"k": 1.5}, "mad": {"k": 3.5},
}

RESULT_COLUMNS = [
    "dataset", "model", "input_type", "horizon", "model_seed", "aggregation_method",
    "anomaly_type", "injection_seed", "anomaly_ratio", "magnitude_scale",
    "detector_type", "threshold_method", "threshold_value",
    "tp", "fp", "tn", "fn", "precision", "recall", "f1",
    "false_positive_rate", "true_negative_rate", "num_points",
    "num_true_anomaly", "num_predicted_anomaly",
    "num_true_events", "num_detected_events", "event_recall",
    "mean_detection_delay", "median_detection_delay", "max_detection_delay",
]


def best_model_for(best_df, dataset, horizon):
    row = best_df[(best_df.dataset == dataset) & (best_df.input_type == INPUT_TYPE)
                  & (best_df.horizon == horizon)]
    if row.empty:
        raise ValueError(f"No best model for {dataset}/h{horizon}.")
    return row.iloc[0]["best_model"]


def experiment_id(dataset, model, horizon):
    return f"{dataset}_{model}_{INPUT_TYPE}_len96_h{horizon}_seed{MODEL_SEED}"


def detector_scores(detector_type, injected_df, val_df):
    """Thin wrapper over src.anomaly.score_detector (kept for compatibility)."""
    return score_detector(detector_type, injected_df, val_df,
                          rolling_window=ROLLING_WINDOW)


def evaluate(detector_type, injected_df, val_df, threshold_method):
    """Threshold from validation, detect on injected test, compute metrics."""
    val_scores, test_scores = detector_scores(detector_type, injected_df, val_df)
    threshold = compute_threshold(val_scores, threshold_method,
                                  **THRESHOLD_PARAMS[threshold_method])
    scored = injected_df.copy()
    scored["detector_score"] = test_scores
    detected = detect_anomalies(scored, threshold, score_col="detector_score")
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
                    for inj_seed in INJECTION_SEEDS:
                        # One injected set shared by all detectors (fair comparison).
                        injected = inject_synthetic_anomalies(
                            test, anomaly_type, anomaly_ratio=ANOMALY_RATIO,
                            duration_range=DURATION_RANGES[anomaly_type],
                            magnitude_scale=MAGNITUDE_SCALE, seed=inj_seed)
                        for detector in DETECTORS:
                            for tmethod in THRESHOLD_METHODS:
                                detected, m = evaluate(detector, injected, val, tmethod)
                                event_m = calculate_event_detection_metrics(detected)
                                rows.append({
                                    "dataset": dataset, "model": model,
                                    "input_type": INPUT_TYPE, "horizon": horizon,
                                    "model_seed": MODEL_SEED, "aggregation_method": agg,
                                    "anomaly_type": anomaly_type, "injection_seed": inj_seed,
                                    "anomaly_ratio": ANOMALY_RATIO,
                                    "magnitude_scale": MAGNITUDE_SCALE,
                                    "detector_type": detector, "threshold_method": tmethod,
                                    **m, **event_m,
                                })
                                # Representative figures (residual detector, seed 42).
                                if (detector == "residual" and dataset == "ETTh1"
                                        and horizon == 24 and agg == "first"
                                        and tmethod == "percentile" and inj_seed == 42):
                                    # residual detector: detector_score == anomaly_score,
                                    # so the existing anomaly_score column is correct.
                                    plot_anomaly_detection(
                                        detected,
                                        os.path.join(fig_dir, f"ETTh1_h24_first_{anomaly_type}_percentile99.png"),
                                        title=f"ETTh1 {model} h24 first — {anomaly_type} (residual, percentile99)")

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    out = os.path.join(results_dir, "anomaly", "metrics",
                       "anomaly_detection_results_v3.csv")
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
