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
import shutil
from collections import defaultdict
from pathlib import Path

# --- paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
METRICS = RESULTS / "metrics"
ANOM = RESULTS / "anomaly" / "metrics"
PRED_DIR = RESULTS / "predictions"
EXTERNAL = RESULTS / "external_validity"
SENSITIVITY = RESULTS / "sensitivity"
FIGURES = RESULTS / "figures"
OUT = REPO_ROOT / "web" / "public" / "data"
FIG_OUT = REPO_ROOT / "web" / "public" / "figures"

# Committed result figures shown as static images on the site (no CSV exists
# for these). Copied into public/figures/ so Vite serves them.
COPY_FIGURES = [
    "attention_by_lag.png",
    "attention_last_layer_heatmap.png",
    "eda_ot_trend.png",
    "eda_ot_distribution.png",
    "eda_feature_timeseries.png",
    "eda_correlation_heatmap.png",
]

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
    # The prediction source CSVs are heavy and git-ignored, so they are absent in
    # CI. Only rewrite when we actually regenerated from source — otherwise leave
    # the committed prediction JSON (public/data/predictions/) untouched.
    if index:
        write_json("predictions/index.json", index)
    else:
        print("  predictions: source CSVs absent — keeping committed JSON")


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

    # stress test: unify the original (spike/level_shift/frozen) and extended
    # (drift/noise_burst/stuck_with_jitter) detector benchmarks into one
    # (anomaly_type, detector) matrix. The two experiment sets report different
    # metric families, so cells are intentionally sparse — a missing metric is
    # left as null rather than filled, to stay faithful to what was measured.
    stress: dict[tuple[str, str], dict] = defaultdict(dict)
    for r in read_csv(ANOM / "anomaly_threshold_free_summary.csv"):
        stress[(r["anomaly_type"], r["detector_type"])].update(
            pr_auc=num(r["mean_pr_auc"]),
            roc_auc=num(r["mean_roc_auc"]),
            best_f1=num(r["mean_best_f1"]),
        )
    for r in read_csv(ANOM / "anomaly_extended_types_summary.csv"):
        stress[(r["anomaly_type"], r["detector_type"])].update(
            pr_auc=num(r["mean_average_precision"]),
            best_f1=num(r["mean_oracle_best_f1"]),
            event_recall=num(r["mean_event_recall"]),
            detection_delay=num(r["mean_detection_delay"], 1),
        )
    payload["stress_test"] = [
        {
            "anomaly_type": atype,
            "detector_type": dtype,
            "pr_auc": m.get("pr_auc"),
            "roc_auc": m.get("roc_auc"),
            "best_f1": m.get("best_f1"),
            "event_recall": m.get("event_recall"),
            "detection_delay": m.get("detection_delay"),
        }
        for (atype, dtype), m in sorted(stress.items())
    ]

    write_json("anomaly.json", payload)


# --- 4. efficiency: accuracy vs model complexity --------------------------
def build_efficiency() -> None:
    rows = read_csv(METRICS / "efficiency_complexity_summary.csv")
    records = [
        {
            "dataset": r["dataset"],
            "model": r["model"],
            "horizon": int(r["horizon"]),
            "input_type": r["input_type"],
            "mae": num(r["mean_mae"]),
            "mae_std": num(r["std_mae"]),
            "params": int(r["total_parameters"]),
            "checkpoint_mb": num(r["checkpoint_size_mb"], 3),
            "epochs": num(r["mean_epochs_ran"], 1),
        }
        for r in rows
    ]
    write_json("efficiency.json", records)


# --- 5. frozen failure case: the residual blind spot + flatness fix -------
def build_frozen() -> None:
    det_order = ["residual", "flatness", "hybrid_or"]
    det_label = {"residual": "Residual", "flatness": "Flatness", "hybrid_or": "Hybrid (OR)"}
    by_det = {
        r["detector_type"]: r
        for r in read_csv(ANOM / "anomaly_hybrid_summary_by_detector.csv")
        if r["anomaly_type"] == "frozen"
    }
    detectors = []
    for d in det_order:
        r = by_det.get(d)
        if not r:
            continue
        detectors.append(
            {
                "name": d,
                "label": det_label[d],
                "f1": num(r["mean_f1"]),
                "recall": num(r["mean_recall"]),
                "precision": num(r["mean_precision"]),
                "event_recall": num(r["mean_event_recall"]),
            }
        )

    # why residuals fail: signal separation (anomaly/normal), mean over configs
    diag = read_csv(ANOM / "frozen_flatness_diagnostics.csv")
    fr = [float(r["ratio_anomaly_to_normal"]) for r in diag]
    rr = [float(r["residual_ratio_anomaly_to_normal"]) for r in diag]
    diagnosis = {
        "flatness_ratio": round(sum(fr) / len(fr), 1),
        "residual_ratio": round(sum(rr) / len(rr), 2),
    }

    # contrast: the same residual detector is strong on the other anomaly types
    by_type = {r["anomaly_type"]: r for r in read_csv(ANOM / "anomaly_summary_by_type.csv")}
    contrast = {
        t: num(by_type[t]["best_mean_f1"]) for t in ("spike", "level_shift") if t in by_type
    }

    write_json("frozen.json", {"detectors": detectors, "diagnosis": diagnosis, "contrast": contrast})


# --- external validity: does the finding transfer to 15-minute ETTm? ------
def build_ettm() -> None:
    rows = read_csv(EXTERNAL / "ettm_forecasting_comparison.csv")
    records = [
        {
            "dataset": r["dataset"],
            "model": r["model"],
            "horizon": int(r["horizon"]),
            "input_type": r["input_type"],
            "mae": num(r["mean_mae"]),
            "mae_std": num(r["std_mae"]),
            "rmse": num(r["mean_rmse"]),
            "wape": num(r["mean_wape"], 2),
            "params": int(r["total_parameters"]),
            "rank": int(r["rank_within_dataset"]),
        }
        for r in rows
    ]
    write_json("ettm.json", records)


# --- input-length sensitivity ablation ------------------------------------
def build_inputlen() -> None:
    rows = read_csv(SENSITIVITY / "input_len_ablation.csv")
    records = [
        {
            "model": r["model"],
            "input_len": int(r["input_len"]),
            "mae": num(r["mean_mae"]),
            "mae_std": num(r["std_mae"]),
            "rmse": num(r["mean_rmse"]),
            "wape": num(r["mean_wape"], 2),
            "params": int(r["total_parameters"]),
            "rank": int(r["rank_within_input_len"]),
        }
        for r in rows
    ]
    summary = [
        {
            "input_len": int(r["input_len"]),
            "best_model": r["best_model"],
            "ranking": r["ranking_best_to_worst"],
        }
        for r in read_csv(SENSITIVITY / "input_len_ablation_summary.csv")
    ]
    write_json("input_length.json", {"rows": records, "summary": summary})


# --- 6. attention analysis (supplementary) + static figures ---------------
def build_attention() -> None:
    rows = read_csv(METRICS / "attention_summary.csv")
    layers = [
        {
            "layer": int(r["layer"]),
            "peak_lag": int(r["peak_lag"]),
            "peak_attention": num(r["peak_attention"]),
            "entropy": num(r["entropy_nats"], 3),
            "max_entropy": num(r["max_entropy_nats"], 3),
            "recent_8_mass": num(r["recent_8_mass"]),
        }
        for r in rows
    ]
    write_json(
        "attention.json",
        {
            "layers": layers,
            "experiment_id": rows[0]["experiment_id"] if rows else None,
            "input_len": int(rows[0]["input_len"]) if rows else 96,
            "figures": {
                "by_lag": "figures/attention_by_lag.png",
                "heatmap": "figures/attention_last_layer_heatmap.png",
            },
        },
    )


def build_figures() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for name in COPY_FIGURES:
        src = FIGURES / name
        if src.exists():
            shutil.copy2(src, FIG_OUT / name)
            print(f"  copied figure {name}")
        else:
            print(f"  skip (missing figure) {name}")


# --- 7. manifest + headline stats -----------------------------------------
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
    build_efficiency()
    build_ettm()
    build_inputlen()
    build_frozen()
    build_attention()
    build_figures()
    build_manifest(comparison)
    print("Done.")


if __name__ == "__main__":
    main()
