"""Hybrid residual + flatness anomaly detection experiment (Module 6.10).

Compares four detectors across ALL anomaly types (not just frozen):

    residual        - forecast-residual magnitude (strong for spike/level_shift)
    flatness        - temporal flatness (targets frozen / stuck sensors)
    hybrid_or       - residual_alarm OR flatness_alarm (deployable rule)
    hybrid_rankmax  - max of validation-percentile ranks (threshold-free score)

All thresholds and rank references come only from validation. The key output is
the trade-off: flatness/hybrid should rescue frozen, potentially at some
precision/FPR cost on spike/level_shift.
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
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.anomaly import (
    inject_synthetic_anomalies, load_prediction_file, score_detector,
    compute_threshold, calculate_detection_metrics,
    calculate_event_detection_metrics, calculate_threshold_free_metrics,
    detect_hybrid_or,
)
from experiments.run_anomaly_detection import (
    best_model_for, experiment_id, DATASETS, HORIZONS, AGGREGATIONS,
    ANOMALY_TYPES, INJECTION_SEEDS, DURATION_RANGES, ANOMALY_RATIO,
    THRESHOLD_METHODS, THRESHOLD_PARAMS,
)

SCORE_DETECTORS = ["residual", "flatness", "hybrid_rankmax"]
ALL_DETECTORS = ["residual", "flatness", "hybrid_or", "hybrid_rankmax"]


def _metrics(injected, predicted):
    df = injected.copy()
    df["predicted_anomaly"] = np.asarray(predicted).astype(bool)
    return {**calculate_detection_metrics(df), **calculate_event_detection_metrics(df)}


def run(project_root, results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    residual_dir = os.path.join(results_dir, "anomaly", "residuals")
    metrics_dir = os.path.join(results_dir, "anomaly", "metrics")
    fig_dir = os.path.join(results_dir, "anomaly", "figures")
    best_df = pd.read_csv(os.path.join(
        results_dir, "metrics", "best_model_by_dataset_horizon.csv"))

    point_rows, tf_rows = [], []
    pr_source = {}

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

                        # Score-based detectors: reusable scores.
                        scores = {d: score_detector(d, injected, val) for d in SCORE_DETECTORS}

                        base = {"dataset": dataset, "model": model, "horizon": horizon,
                                "aggregation_method": agg, "anomaly_type": anomaly_type,
                                "injection_seed": inj_seed}

                        # Threshold-free (score-based detectors only).
                        for d in SCORE_DETECTORS:
                            tf = calculate_threshold_free_metrics(labels, scores[d][1])
                            tf_rows.append({**base, "detector_type": d, **tf})
                            if (dataset == "ETTh1" and horizon == 24 and agg == "first"
                                    and anomaly_type == "frozen" and inj_seed == 42):
                                pr_source[d] = (labels, scores[d][1])

                        # Point + event metrics per threshold method.
                        for tmethod in THRESHOLD_METHODS:
                            params = THRESHOLD_PARAMS[tmethod]
                            thr = {d: compute_threshold(scores[d][0], tmethod, **params)
                                   for d in SCORE_DETECTORS}
                            for d in SCORE_DETECTORS:
                                predicted = scores[d][1] > thr[d]
                                point_rows.append({**base, "detector_type": d,
                                                   "threshold_method": tmethod,
                                                   "threshold_value": thr[d],
                                                   **_metrics(injected, predicted)})
                            # hybrid_or: OR of residual and flatness alarms.
                            predicted_or = detect_hybrid_or(
                                scores["residual"][1], scores["flatness"][1],
                                thr["residual"], thr["flatness"])
                            point_rows.append({**base, "detector_type": "hybrid_or",
                                               "threshold_method": tmethod,
                                               "threshold_value": np.nan,
                                               **_metrics(injected, predicted_or)})

    point_df = pd.DataFrame(point_rows)
    tf_df = pd.DataFrame(tf_rows)
    os.makedirs(metrics_dir, exist_ok=True)
    point_df.to_csv(os.path.join(metrics_dir, "anomaly_hybrid_results.csv"), index=False)
    tf_df.to_csv(os.path.join(metrics_dir, "anomaly_hybrid_threshold_free_results.csv"), index=False)

    _summaries(point_df, tf_df, metrics_dir)
    _figures(point_df, pr_source, fig_dir)
    print(f"Wrote anomaly_hybrid_results.csv ({len(point_df)} rows) and "
          f"anomaly_hybrid_threshold_free_results.csv ({len(tf_df)} rows).")


def _summaries(point_df, tf_df, metrics_dir):
    by_det = point_df.groupby(["detector_type", "anomaly_type"]).agg(
        mean_precision=("precision", "mean"), mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"), mean_fpr=("false_positive_rate", "mean"),
        mean_event_recall=("event_recall", "mean"), num_runs=("f1", "count"),
    ).reset_index()
    by_det.to_csv(os.path.join(metrics_dir, "anomaly_hybrid_summary_by_detector.csv"), index=False)

    # Best detector per anomaly type by mean F1 (point) and mean PR-AUC (threshold-free).
    f1 = point_df.groupby(["anomaly_type", "detector_type"])["f1"].mean().reset_index()
    best_f1 = f1.loc[f1.groupby("anomaly_type")["f1"].idxmax()].rename(
        columns={"detector_type": "best_f1_detector", "f1": "best_mean_f1"})
    pra = tf_df.groupby(["anomaly_type", "detector_type"])["pr_auc"].mean().reset_index()
    best_pra = pra.loc[pra.groupby("anomaly_type")["pr_auc"].idxmax()].rename(
        columns={"detector_type": "best_pr_auc_detector", "pr_auc": "best_mean_pr_auc"})
    by_type = best_f1.merge(best_pra, on="anomaly_type")
    by_type.to_csv(os.path.join(metrics_dir, "anomaly_hybrid_summary_by_type.csv"), index=False)


def _figures(point_df, pr_source, fig_dir):
    from src.viz import PALETTE, apply_paper_style
    apply_paper_style()
    os.makedirs(fig_dir, exist_ok=True)
    # PR curves for frozen (residual vs flatness vs hybrid_rankmax).
    if pr_source:
        plt.figure(figsize=(8, 6))
        for d, (labels, s) in pr_source.items():
            precision, recall, _ = precision_recall_curve(labels, s)
            plt.plot(recall, precision, label=d)
        plt.title("PR curves — frozen anomaly (ETTh1 h24 first)")
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "hybrid_frozen_pr_curve.png"))
        plt.close()

    # Event recall by detector for frozen (mean over scenarios/seeds/thresholds).
    frozen = point_df[point_df.anomaly_type == "frozen"]
    er = frozen.groupby("detector_type")["event_recall"].mean().reindex(ALL_DETECTORS)
    plt.figure(figsize=(7, 5))
    plt.bar(er.index, er.values,
            color=[PALETTE[0], PALETTE[2], PALETTE[4], PALETTE[1]])
    plt.title("Frozen event recall by detector")
    plt.ylabel("mean event recall")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hybrid_frozen_event_recall.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hybrid anomaly detection experiment.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()
    run(args.project_root)


if __name__ == "__main__":
    main()
