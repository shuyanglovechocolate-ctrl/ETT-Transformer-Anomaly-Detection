"""Anomaly magnitude sensitivity control experiment (Module 6.7).

Tests whether the residual detector's advantage over simple baselines is robust
across anomaly magnitudes, or an artifact of a single magnitude choice. Sweeps
``magnitude_scale`` and records F1 for every detector, so the single-point
comparison of Module 6.6 becomes a curve.

Pure post-processing (no re-training); reuses the Module 6.5/6.6 detection chain.
Frozen anomalies are magnitude-independent, so only spike / level_shift are swept.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import inject_synthetic_anomalies, load_prediction_file
from experiments.run_anomaly_detection import (
    best_model_for, experiment_id, evaluate,
    DATASETS, HORIZONS, DETECTORS, DURATION_RANGES,
    ANOMALY_RATIO, INJECTION_SEEDS,
)

MAGNITUDE_SCALES = [1.0, 2.0, 3.0, 5.0]
SWEEP_ANOMALY_TYPES = ["spike", "level_shift"]  # frozen is magnitude-independent
AGGREGATION = "first"          # fixed to the tightest-residual setting
THRESHOLD_METHOD = "percentile"  # fixed representative threshold

RESULT_COLUMNS = [
    "dataset", "model", "horizon", "aggregation_method", "anomaly_type",
    "magnitude_scale", "injection_seed", "detector_type", "threshold_method",
    "precision", "recall", "f1", "false_positive_rate",
]


def run(project_root, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    residual_dir = os.path.join(results_dir, "anomaly", "residuals")
    best_df = pd.read_csv(os.path.join(
        results_dir, "metrics", "best_model_by_dataset_horizon.csv"))

    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            model = best_model_for(best_df, dataset, horizon)
            eid = experiment_id(dataset, model, horizon)
            val = load_prediction_file(
                os.path.join(residual_dir, f"{eid}_val_residual_{AGGREGATION}.csv"))
            test = load_prediction_file(
                os.path.join(residual_dir, f"{eid}_test_residual_{AGGREGATION}.csv"))
            for anomaly_type in SWEEP_ANOMALY_TYPES:
                for magnitude in MAGNITUDE_SCALES:
                    for inj_seed in INJECTION_SEEDS:
                        injected = inject_synthetic_anomalies(
                            test, anomaly_type, anomaly_ratio=ANOMALY_RATIO,
                            duration_range=DURATION_RANGES[anomaly_type],
                            magnitude_scale=magnitude, seed=inj_seed)
                        for detector in DETECTORS:
                            _, m = evaluate(detector, injected, val, THRESHOLD_METHOD)
                            rows.append({
                                "dataset": dataset, "model": model, "horizon": horizon,
                                "aggregation_method": AGGREGATION,
                                "anomaly_type": anomaly_type,
                                "magnitude_scale": magnitude, "injection_seed": inj_seed,
                                "detector_type": detector,
                                "threshold_method": THRESHOLD_METHOD,
                                "precision": m["precision"], "recall": m["recall"],
                                "f1": m["f1"],
                                "false_positive_rate": m["false_positive_rate"],
                            })

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    out = os.path.join(results_dir, "anomaly", "metrics",
                       "anomaly_magnitude_sensitivity.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results.to_csv(out, index=False)
    print(f"Wrote {out} ({len(results)} rows)")

    _plot(results, os.path.join(results_dir, "anomaly", "figures"))
    return results


def _plot(results, fig_dir):
    """F1 vs magnitude by detector (mean over datasets/horizons/seeds), per type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.viz import apply_paper_style

    apply_paper_style()
    os.makedirs(fig_dir, exist_ok=True)
    for anomaly_type in SWEEP_ANOMALY_TYPES:
        sub = results[results.anomaly_type == anomaly_type]
        plt.figure(figsize=(8, 5))
        for detector in DETECTORS:
            d = sub[sub.detector_type == detector].groupby("magnitude_scale")["f1"].mean()
            plt.plot(d.index, d.values, marker="o", label=detector)
        plt.title(f"F1 vs anomaly magnitude — {anomaly_type}")
        plt.xlabel("magnitude_scale (x std of clean OT)")
        plt.ylabel("mean F1")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"magnitude_sensitivity_{anomaly_type}.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Anomaly magnitude sensitivity sweep.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
