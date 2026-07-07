"""Extended synthetic anomaly types — construct-validity stress test (Module 7.7).

This is a SEPARATE, additive experiment. It does not touch the core three-type
canonical results (spike / level_shift / frozen), the threshold-free table, the
hybrid results, or the significance tests that back the v1.0-thesis release.

It probes whether the residual / flatness / causal-baseline detectors generalise
beyond the idealised core anomalies to three more fault-like synthetic types:

- drift             : slow linear ramp (gradual sensor drift).
- noise_burst       : short high-variance noise burst (measurement instability).
- stuck_with_jitter : frozen plus small jitter (a stuck sensor that is not perfectly
                      constant) — a direct stress test of the flatness signal.

These extended types are NOT claimed to be real fault labels; they are additional
synthetic stress tests. Detection reuses the on-disk aggregated residual files and
the shared scoring dispatch (no re-training). Reported per (scenario, detector):
threshold-free PR-AUC / ROC-AUC / oracle best-F1, plus event-wise recall and mean
detection delay (validation percentile-99 threshold, as in the core pipeline).

Reads  results/anomaly/residuals/*.csv  (regenerable, git-ignored)
Writes results/anomaly/metrics/anomaly_extended_types_results.csv
       results/anomaly/metrics/anomaly_extended_types_summary.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    inject_synthetic_anomalies, load_prediction_file, score_detector,
    calculate_threshold_free_metrics, calculate_event_detection_metrics,
    compute_threshold,
)
from experiments.run_anomaly_detection import (
    best_model_for, experiment_id, DATASETS, HORIZONS, AGGREGATIONS,
    INJECTION_SEEDS, ANOMALY_RATIO,
)

EXTENDED_ANOMALY_TYPES = ["drift", "noise_burst", "stuck_with_jitter"]
EXTENDED_DETECTORS = ["residual", "raw_zscore", "diff_score",
                      "rolling_zscore", "flatness"]
# Slow drift is given longer segments; a noise burst is short-to-medium; a stuck
# sensor lasts as long as the core frozen anomaly.
EXTENDED_DURATION_RANGES = {
    "drift": (24, 48),
    "noise_burst": (6, 12),
    "stuck_with_jitter": (12, 24),
}
EVENT_PERCENTILE = 99

RESULT_COLUMNS = [
    "dataset", "model", "horizon", "aggregation_method", "anomaly_type",
    "injection_seed", "detector_type",
    "average_precision", "roc_auc", "oracle_best_f1",
    "event_recall", "mean_detection_delay",
]


def evaluate_detector(injected: pd.DataFrame, val: pd.DataFrame, detector: str,
                      percentile: float = EVENT_PERCENTILE) -> dict:
    """Threshold-free + event-wise metrics for one detector on one injected series.

    Alarms for the event metrics use a validation percentile threshold; they are
    computed as ``test_scores > threshold`` directly so the routine works for every
    detector regardless of score sign convention.
    """
    labels = injected["is_anomaly"].to_numpy()
    val_scores, test_scores = score_detector(detector, injected, val)
    tf = calculate_threshold_free_metrics(labels, test_scores)

    threshold = compute_threshold(np.asarray(val_scores, dtype=float),
                                  "percentile", percentile=percentile)
    preds = np.asarray(test_scores, dtype=float) > threshold
    work = pd.DataFrame({
        "is_anomaly": labels,
        "anomaly_segment_id": injected["anomaly_segment_id"].to_numpy(),
        "predicted_anomaly": preds,
    })
    event = calculate_event_detection_metrics(work)

    return {
        "average_precision": tf["pr_auc"],
        "roc_auc": tf["roc_auc"],
        "oracle_best_f1": tf["best_f1"],
        "event_recall": event["event_recall"],
        "mean_detection_delay": event["mean_detection_delay"],
    }


def run(project_root, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    residual_dir = os.path.join(results_dir, "anomaly", "residuals")
    metrics_dir = os.path.join(results_dir, "anomaly", "metrics")
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
                for anomaly_type in EXTENDED_ANOMALY_TYPES:
                    for inj_seed in INJECTION_SEEDS:
                        injected = inject_synthetic_anomalies(
                            test, anomaly_type, anomaly_ratio=ANOMALY_RATIO,
                            duration_range=EXTENDED_DURATION_RANGES[anomaly_type],
                            seed=inj_seed)
                        for detector in EXTENDED_DETECTORS:
                            m = evaluate_detector(injected, val, detector)
                            rows.append({
                                "dataset": dataset, "model": model,
                                "horizon": horizon, "aggregation_method": agg,
                                "anomaly_type": anomaly_type,
                                "injection_seed": inj_seed,
                                "detector_type": detector, **m})

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    os.makedirs(metrics_dir, exist_ok=True)
    results.to_csv(os.path.join(metrics_dir, "anomaly_extended_types_results.csv"),
                   index=False)

    summary = results.groupby(["anomaly_type", "detector_type"]).agg(
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
        mean_oracle_best_f1=("oracle_best_f1", "mean"),
        mean_event_recall=("event_recall", "mean"),
        mean_detection_delay=("mean_detection_delay", "mean"),
        num_runs=("average_precision", "count"),
    ).reset_index()
    summary.to_csv(os.path.join(metrics_dir, "anomaly_extended_types_summary.csv"),
                   index=False)
    print(f"Wrote anomaly_extended_types_results.csv ({len(results)} rows) and summary.")


def main():
    parser = argparse.ArgumentParser(
        description="Extended synthetic anomaly-type stress test (post-hoc).")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
