"""Prepare validation and test residuals for anomaly detection (Module 6.1).

For each anomaly-detection scenario (best model per dataset x horizon), this
loads the saved best checkpoint and config snapshot, runs inference on the
VALIDATION set (never saved during Module 3), and aggregates both validation and
test residuals into per-timestamp anomaly scores.

Thresholds (Module 6.3) will be estimated from the VALIDATION residuals, so the
test set is never used to set thresholds — avoiding leakage.

No model is re-trained; this is inference only.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.data.pipeline import build_data_pipeline  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402
from src.training.predictor import predict, create_prediction_dataframe  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.anomaly.residuals import (  # noqa: E402
    aggregate_residuals, load_prediction_file, save_aggregated_residuals,
    AGGREGATION_METHODS,
)

DATASETS = ["ETTh1", "ETTh2"]
HORIZONS = [24, 96]
INPUT_TYPE = "multivariate"
SEED = 42


def best_model_for(best_df, dataset, input_type, horizon):
    row = best_df[(best_df.dataset == dataset) & (best_df.input_type == input_type)
                  & (best_df.horizon == horizon)]
    if row.empty:
        raise ValueError(f"No best model for {dataset}/{input_type}/h{horizon}.")
    return row.iloc[0]["best_model"]


def experiment_id(dataset, model, input_type, horizon, seed):
    return f"{dataset}_{model}_{input_type}_len96_h{horizon}_seed{seed}"


def prepare_scenario(dataset, horizon, best_df, project_root, device):
    model_name = best_model_for(best_df, dataset, INPUT_TYPE, horizon)
    eid = experiment_id(dataset, model_name, INPUT_TYPE, horizon, SEED)

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

    # Validation predictions (generated fresh; used to set thresholds).
    yv_true, yv_pred = predict(model, data["val_loader"], device)
    val_df = create_prediction_dataframe(yv_true, yv_pred, data["val_y_dates"],
                                         data["scaler_y"])

    # Test predictions (reuse the file saved in Module 3).
    test_path = os.path.join(results, "predictions", f"{eid}_predictions.csv")
    test_df = load_prediction_file(test_path)

    out_dir = os.path.join(results, "anomaly", "residuals")
    for method in AGGREGATION_METHODS:
        save_aggregated_residuals(
            aggregate_residuals(val_df, method),
            os.path.join(out_dir, f"{eid}_val_residual_{method}.csv"))
        save_aggregated_residuals(
            aggregate_residuals(test_df, method),
            os.path.join(out_dir, f"{eid}_test_residual_{method}.csv"))

    print(f"[{eid}] val/test residuals aggregated ({', '.join(AGGREGATION_METHODS)}).")
    return eid


def main():
    parser = argparse.ArgumentParser(description="Prepare anomaly residuals.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")
    best_df = pd.read_csv(os.path.join(
        args.project_root, "results", "metrics", "best_model_by_dataset_horizon.csv"))

    for dataset in DATASETS:
        for horizon in HORIZONS:
            prepare_scenario(dataset, horizon, best_df, args.project_root, device)


if __name__ == "__main__":
    main()
