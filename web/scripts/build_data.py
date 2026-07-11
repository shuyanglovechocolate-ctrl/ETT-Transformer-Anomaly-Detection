#!/usr/bin/env python3
"""Build static JSON assets for the ETT showcase frontend.

Reads the thesis result artifacts under ``results/`` and emits compact JSON
files under ``web/public/data/`` that the static React site fetches at runtime.
The thesis outputs are static, so this runs once (or in CI) and needs no backend.

Pure standard library (csv + json) so it runs anywhere with no pip installs.

Usage:
    python web/scripts/build_data.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

# --- paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
METRICS = RESULTS / "metrics"
ANOM = RESULTS / "anomaly" / "metrics"
PRED_DIR = RESULTS / "predictions"
OUT = REPO_ROOT / "web" / "public" / "data"

# Representative forecast series to expose in the interactive chart.
# Kept small on purpose: one seed / horizon per (dataset, model, input_type).
PRED_MODELS = ["naive", "linear", "nlinear", "dlinear", "lstm", "transformer"]
PRED_DATASETS = ["ETTh1", "ETTh2"]
PRED_INPUT = "multivariate"
PRED_HORIZON = 24
PRED_SEED = 42
PRED_MAX_POINTS = 800  # downsample target for the browser


def num(x, ndigits=4):
    """Parse to float and round; empty/NaN -> None."""
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return round(v, ndigits)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(name: str, payload) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


# --- 1. forecasting model comparison --------------------------------------
def build_comparison() -> list[dict]:
    rows = read_csv(METRICS / "model_comparison.csv")
    records = [
        {
            "dataset": r["dataset"],
            "model": r["model"],
            "input_type": r["input_type"],
            "horizon": int(r["horizon"]),
            "mae": num(r["mean_mae"]),
            "mae_std": num(r["std_mae"]),
            "rmse": num(r["mean_rmse"]),
            "rmse_std": num(r["std_rmse"]),
            "wape": num(r["mean_wape"], 2),
            "wape_std": num(r["std_wape"], 2),
            "num_runs": int(r["num_runs"]),
        }
        for r in rows
    ]
    write_json("comparison.json", records)
    return records


# --- 2. representative forecast series (y_true vs y_pred) ------------------
def build_predictions() -> None:
    index = []
    for dataset in PRED_DATASETS:
        for model in PRED_MODELS:
            stem = (
                f"{dataset}_{model}_{PRED_INPUT}_len96_h{PRED_HORIZON}"
                f"_seed{PRED_SEED}_predictions.csv"
            )
            path = PRED_DIR / stem
            if not path.exists():
                print(f"  skip (missing) {stem}")
                continue
            rows = [r for r in read_csv(path) if r["horizon_index"] == "0"]
            if len(rows) > PRED_MAX_POINTS:
                step = math.ceil(len(rows) / PRED_MAX_POINTS)
                rows = rows[::step]
            key = f"{dataset}_{model}"
            series = {
                "dataset": dataset,
                "model": model,
                "horizon": PRED_HORIZON,
                "dates": [r["target_date"] for r in rows],
                "y_true": [num(r["y_true"], 3) for r in rows],
                "y_pred": [num(r["y_pred"], 3) for r in rows],
            }
            write_json(f"predictions/{key}.json", series)
            index.append({"key": key, "dataset": dataset, "model": model})
    write_json("predictions/index.json", index)


# --- 3. anomaly detection --------------------------------------------------
def build_anomaly() -> None:
    payload = {}

    payload["by_type"] = [
        {
            "anomaly_type": r["anomaly_type"],
            "best_detector": r["best_detector"],
            "best_mean_f1": num(r["best_mean_f1"]),
            "mean_recall": num(r["mean_recall"]),
            "mean_precision": num(r["mean_precision"]),
        }
        for r in read_csv(ANOM / "anomaly_summary_by_type.csv")
    ]

    payload["threshold_free"] = [
        {
            "detector_type": r["detector_type"],
            "anomaly_type": r["anomaly_type"],
            "pr_auc": num(r["mean_pr_auc"]),
            "pr_auc_std": num(r["std_pr_auc"]),
            "roc_auc": num(r["mean_roc_auc"]),
            "best_f1": num(r["mean_best_f1"]),
            "num_runs": int(r["num_runs"]),
        }
        for r in read_csv(ANOM / "anomaly_threshold_free_summary.csv")
    ]

    payload["accuracy_vs_detection"] = [
        {
            "dataset": r["dataset"],
            "model": r["model"],
            "horizon": int(r["horizon"]),
            "anomaly_type": r["anomaly_type"],
            "forecast_mae": num(r["forecast_mae"], 3),
            "average_precision": num(r["average_precision"]),
            "event_recall": num(r["event_recall"]),
        }
        for r in read_csv(ANOM / "accuracy_vs_detection.csv")
    ]

    # magnitude sensitivity: mean f1 vs magnitude scale (residual detector only)
    agg = defaultdict(list)
    for r in read_csv(ANOM / "anomaly_magnitude_sensitivity.csv"):
        if r["detector_type"] != "residual":
            continue
        agg[(r["anomaly_type"], float(r["magnitude_scale"]))].append(float(r["f1"]))
    payload["magnitude_sensitivity"] = [
        {
            "anomaly_type": atype,
            "magnitude_scale": round(scale, 2),
            "f1": round(sum(vals) / len(vals), 4),
        }
        for (atype, scale), vals in sorted(agg.items())
    ]

    write_json("anomaly.json", payload)


# --- 4. manifest + headline stats -----------------------------------------
def build_manifest(comparison: list[dict]) -> None:
    manifest = json.loads((METRICS / "reproducibility_manifest.json").read_text())

    maes = [r["mae"] for r in comparison if r["mae"] is not None]
    headline = {
        "best_mae": round(min(maes), 3),
        "num_models": len({r["model"] for r in comparison}),
        "num_datasets": len({r["dataset"] for r in comparison}),
        "num_horizons": len({r["horizon"] for r in comparison}),
        "num_seeds": max(r["num_runs"] for r in comparison),
        "num_anomaly_types": 3,
    }

    write_json("manifest.json", {"headline": headline, "reproducibility": manifest})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building showcase data ->", OUT.relative_to(REPO_ROOT))
    comparison = build_comparison()
    build_predictions()
    build_anomaly()
    build_manifest(comparison)
    print("Done.")


if __name__ == "__main__":
    main()
