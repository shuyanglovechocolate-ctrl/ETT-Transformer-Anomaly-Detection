"""Inference-latency analysis (Module 8.4).

Measures forward-pass (inference) latency for each model on the ETTh1 test loader, as
a deployability complement to the parameter/checkpoint-size efficiency table (7.6).
This is a SEPARATE table; it does not modify efficiency_complexity_summary.csv.

Latency is determined by architecture and input size, not by the trained weight
values, so models are built fresh (no checkpoint loading) — this keeps the measurement
reproducible on a clean clone. Training wall-clock time was not recorded historically
and is deliberately not reported here (reporting it would require re-timed retraining).

Writes results/metrics/inference_latency_summary.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODELS = ["naive", "linear", "nlinear", "dlinear", "lstm", "transformer"]
DATASET, INPUT_TYPE, INPUT_LEN, HORIZON, SEED = "ETTh1", "multivariate", 96, 96, 42

SUMMARY_COLS = ["dataset", "model", "horizon", "input_type", "device",
                "num_windows", "num_batches", "num_repeats", "parameters",
                "mean_latency_ms_per_batch", "mean_latency_ms_per_1k_windows",
                "std_latency_ms_per_1k_windows"]


def summarize_latency(repeat_seconds, num_windows: int, num_batches: int) -> dict:
    """Convert per-pass wall-clock seconds into per-batch / per-1k-window latencies."""
    arr = np.asarray(repeat_seconds, dtype=float)
    per_1k = arr / num_windows * 1e6          # ms per 1000 windows
    per_batch = arr / num_batches * 1e3       # ms per batch
    return {
        "num_repeats": int(len(arr)),
        "mean_latency_ms_per_batch": float(per_batch.mean()),
        "mean_latency_ms_per_1k_windows": float(per_1k.mean()),
        "std_latency_ms_per_1k_windows": float(per_1k.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def _sync(device):
    import torch
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_model(model, loader, device, warmup: int = 2, repeats: int = 20):
    """Time full forward passes over `loader`; returns (per_pass_seconds, windows, batches)."""
    import torch
    model.eval()
    num_windows = int(sum(x.shape[0] for x, _ in loader))
    num_batches = int(len(loader))
    with torch.no_grad():
        for _ in range(warmup):
            for x, _ in loader:
                model(x.to(device))
            _sync(device)
        repeat_seconds = []
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            for x, _ in loader:
                model(x.to(device))
            _sync(device)
            repeat_seconds.append(time.perf_counter() - t0)
    return repeat_seconds, num_windows, num_batches


def measure(models=MODELS, warmup=2, repeats=20) -> pd.DataFrame:
    import torch  # noqa: F401  (ensures torch import errors surface early)
    from src.data.pipeline import build_data_pipeline
    from src.models import build_model
    from src.utils.device import get_device
    from experiments.run_matrix import build_matrix_config

    device = get_device()
    # The data pipeline is independent of the model, so build it once and reuse it.
    base_config = build_matrix_config(DATASET, INPUT_TYPE, INPUT_LEN, HORIZON, SEED, "naive")
    data = build_data_pipeline(base_config)
    loader = data["test_loader"]

    rows = []
    for model_name in models:
        config = build_matrix_config(DATASET, INPUT_TYPE, INPUT_LEN, HORIZON, SEED, model_name)
        model = build_model(config, data["num_features"], data["feature_cols"]).to(device)
        params = int(sum(p.numel() for p in model.parameters()))
        rep_s, nw, nb = time_model(model, loader, device, warmup=warmup, repeats=repeats)
        rows.append({
            "dataset": DATASET, "model": model_name, "horizon": HORIZON,
            "input_type": INPUT_TYPE, "device": str(device),
            "num_windows": nw, "num_batches": nb, "parameters": params,
            **summarize_latency(rep_s, nw, nb),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLS)


def main():
    parser = argparse.ArgumentParser(description="Inference-latency analysis.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    df = measure(warmup=args.warmup, repeats=args.repeats)
    out = os.path.join(args.results_dir, "metrics", "inference_latency_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    print(df[["model", "parameters", "mean_latency_ms_per_1k_windows"]].to_string(index=False))


if __name__ == "__main__":
    main()
