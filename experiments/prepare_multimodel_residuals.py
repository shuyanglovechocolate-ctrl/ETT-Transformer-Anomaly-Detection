"""Prepare validation/test residuals for MULTIPLE models (Module 6.11).

For the accuracy-vs-detection analysis we need residuals from several forecasters
(not just the best one) on a fixed scenario set. This loads each saved best
checkpoint, runs VALIDATION inference (never re-training), and aggregates val/test
residuals with the ``first`` (one-step) aggregation. The target series y_true is
model-independent, so anomaly injection later uses identical positions across
models automatically.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.pipeline import build_data_pipeline
from src.models import build_model, count_parameters
from src.training.predictor import predict, create_prediction_dataframe
from src.training.checkpoint import load_checkpoint
from src.anomaly import aggregate_residuals, load_prediction_file, save_aggregated_residuals

MODELS = ["naive", "linear", "nlinear", "dlinear", "lstm", "transformer"]
DATASETS = ["ETTh1", "ETTh2"]
HORIZONS = [24, 96]
INPUT_TYPE = "multivariate"
SEED = 42
AGGREGATION = "first"


def experiment_id(dataset, model, horizon):
    return f"{dataset}_{model}_{INPUT_TYPE}_len96_h{horizon}_seed{SEED}"


def build_multimodel_scenarios():
    """Return the (dataset, model, horizon) tuples to prepare."""
    return [(d, m, h) for d in DATASETS for h in HORIZONS for m in MODELS]


def generate(dataset, model_name, horizon, project_root, device, out_dir):
    eid = experiment_id(dataset, model_name, horizon)
    results = os.path.join(project_root, "results")
    config = load_config(os.path.join(results, "logs", f"{eid}_config.yaml"))
    set_seed(config["training"]["seed"])

    data = build_data_pipeline(config, project_root=project_root)
    model = build_model(config, num_features=data["num_features"],
                        feature_cols=data["feature_cols"]).to(device)
    ckpt = os.path.join(results, "checkpoints", f"{eid}_best.pt")
    if count_parameters(model) > 0 and os.path.exists(ckpt):
        load_checkpoint(model, ckpt)
        model.to(device)

    # Validation residuals (fresh inference).
    yv_true, yv_pred = predict(model, data["val_loader"], device)
    val_df = create_prediction_dataframe(yv_true, yv_pred, data["val_y_dates"],
                                         data["scaler_y"])
    save_aggregated_residuals(
        aggregate_residuals(val_df, AGGREGATION),
        os.path.join(out_dir, f"{eid}_val_{AGGREGATION}.csv"))

    # Test residuals (reuse Module 3 prediction file).
    test_df = load_prediction_file(
        os.path.join(results, "predictions", f"{eid}_predictions.csv"))
    save_aggregated_residuals(
        aggregate_residuals(test_df, AGGREGATION),
        os.path.join(out_dir, f"{eid}_test_{AGGREGATION}.csv"))
    print(f"[{eid}] residuals prepared.")


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-model residuals.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")
    out_dir = os.path.join(args.project_root, "results", "anomaly", "multimodel_residuals")
    for dataset, model_name, horizon in build_multimodel_scenarios():
        generate(dataset, model_name, horizon, args.project_root, device, out_dir)


if __name__ == "__main__":
    main()
