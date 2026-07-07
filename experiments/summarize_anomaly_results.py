"""Summarize anomaly detection results into compact tables (Module 6.6).

Reads results/anomaly/metrics/anomaly_detection_results_v3.csv and writes:
- anomaly_summary_by_detector.csv   (detector x anomaly_type mean/std metrics)
- anomaly_summary_by_type.csv        (best detector per anomaly type)
- anomaly_summary_by_threshold.csv   (threshold-method precision/recall tradeoff)
- anomaly_summary_by_horizon.csv     (horizon comparison)
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS = ["precision", "recall", "f1", "false_positive_rate"]


def _mean_std(df, group_cols):
    agg = {}
    for m in METRICS:
        agg[f"mean_{m}"] = (m, "mean")
        agg[f"std_{m}"] = (m, "std")
    agg["num_runs"] = ("f1", "count")
    return df.groupby(group_cols).agg(**agg).reset_index()


def summary_by_detector(df):
    return _mean_std(df, ["detector_type", "anomaly_type"])


def summary_by_type(df):
    per = _mean_std(df, ["anomaly_type", "detector_type"])
    best_idx = per.groupby("anomaly_type")["mean_f1"].idxmax()
    best = per.loc[best_idx, ["anomaly_type", "detector_type", "mean_f1",
                              "mean_recall", "mean_precision"]]
    return best.rename(columns={"detector_type": "best_detector",
                                "mean_f1": "best_mean_f1"}).reset_index(drop=True)


def summary_by_threshold(df):
    return _mean_std(df, ["threshold_method"])


def summary_by_horizon(df):
    return _mean_std(df, ["horizon"])


def event_summary_by_detector(df):
    """Event-wise recall and detection delay per detector x anomaly type."""
    if "event_recall" not in df.columns:
        return None
    return df.groupby(["detector_type", "anomaly_type"]).agg(
        mean_event_recall=("event_recall", "mean"),
        std_event_recall=("event_recall", "std"),
        mean_detection_delay=("mean_detection_delay", "mean"),
        num_runs=("event_recall", "count"),
    ).reset_index()


def event_summary_by_type(df):
    """Best detector per anomaly type by mean event recall."""
    if "event_recall" not in df.columns:
        return None
    per = df.groupby(["anomaly_type", "detector_type"]).agg(
        mean_event_recall=("event_recall", "mean"),
        mean_detection_delay=("mean_detection_delay", "mean"),
    ).reset_index()
    best_idx = per.groupby("anomaly_type")["mean_event_recall"].idxmax()
    best = per.loc[best_idx].rename(columns={"detector_type": "best_detector"})
    return best.reset_index(drop=True)


def summarize(results_path, output_dir):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")
    df = pd.read_csv(results_path)
    os.makedirs(output_dir, exist_ok=True)

    tables = {
        "anomaly_summary_by_detector": summary_by_detector(df),
        "anomaly_summary_by_type": summary_by_type(df),
        "anomaly_summary_by_threshold": summary_by_threshold(df),
        "anomaly_summary_by_horizon": summary_by_horizon(df),
        "anomaly_event_summary_by_detector": event_summary_by_detector(df),
        "anomaly_event_summary_by_type": event_summary_by_type(df),
    }
    tables = {k: v for k, v in tables.items() if v is not None}
    for name, table in tables.items():
        path = os.path.join(output_dir, f"{name}.csv")
        table.to_csv(path, index=False)
        print(f"Wrote {path} ({len(table)} rows)")
    return tables


def main():
    parser = argparse.ArgumentParser(description="Summarize anomaly results.")
    metrics_dir = str(PROJECT_ROOT / "results" / "anomaly" / "metrics")
    parser.add_argument("--results-path",
                        default=os.path.join(metrics_dir, "anomaly_detection_results_v3.csv"))
    parser.add_argument("--output-dir", default=metrics_dir)
    args = parser.parse_args()
    summarize(args.results_path, args.output_dir)


if __name__ == "__main__":
    main()
