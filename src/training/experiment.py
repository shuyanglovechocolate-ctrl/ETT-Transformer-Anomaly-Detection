"""Experiment runner and logging (Module 3.3).

Turns a single training run into a reproducible, recorded experiment: a stable
``experiment_id``, managed output paths, a config snapshot (post-override),
a structured metrics JSON, and an appended row in ``experiment_log.csv``.
"""

import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml

from src.utils.config import load_config, validate_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.pipeline import build_data_pipeline
from src.models import (
    build_model,
    validate_model_config,
    count_parameters,
    get_model_summary,
)
from src.training.trainer import fit_model, build_scheduler
from src.training.evaluator import compute_metrics
from src.training.predictor import predict, create_prediction_dataframe
from src.training.plots import save_loss_curve, save_prediction_plot
from src.training.checkpoint import load_checkpoint


EXPERIMENT_LOG_COLUMNS = [
    "timestamp", "experiment_id", "dataset", "model", "model_type",
    "input_type", "input_len", "horizon", "seed", "num_features",
    "total_parameters", "trainable_parameters",
    "epochs_requested", "epochs_ran", "best_epoch", "best_val_loss",
    "stopped_early", "mae", "rmse", "wape",
    "checkpoint_path", "metrics_path", "predictions_path", "config_path",
]


def build_experiment_id(config: dict) -> str:
    """Build a stable experiment id from the config.

    Format: {dataset}_{model}[_ci]_{input_type}_len{input_len}_h{horizon}_seed{seed}
    The ``_ci`` marker is added for channel-independent DLinear.
    """
    d = config["dataset"]
    w = config["window"]
    model = config["model"]["name"]
    suffix = ""
    if model == "dlinear" and config["model"].get("channel_independent"):
        suffix = "_ci"
    return (
        f"{d['name']}_{model}{suffix}_{d['input_type']}"
        f"_len{w['input_len']}_h{w['horizon']}_seed{config['training']['seed']}"
    )


def build_output_paths(experiment_id: str, results_dir: str = "results") -> Dict[str, str]:
    """Return all output paths for an experiment, keyed by artifact."""
    return {
        "checkpoint": os.path.join(results_dir, "checkpoints", f"{experiment_id}_best.pt"),
        "metrics": os.path.join(results_dir, "metrics", f"{experiment_id}_metrics.json"),
        "predictions": os.path.join(results_dir, "predictions", f"{experiment_id}_predictions.csv"),
        "loss_curve": os.path.join(results_dir, "figures", f"{experiment_id}_loss.png"),
        "prediction_plot": os.path.join(results_dir, "figures", f"{experiment_id}_prediction.png"),
        "history": os.path.join(results_dir, "logs", f"{experiment_id}_history.json"),
        "config_snapshot": os.path.join(results_dir, "logs", f"{experiment_id}_config.yaml"),
    }


def save_config_snapshot(config: dict, path: str) -> None:
    """Save the (post-override) config as YAML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_json(data: dict, path: str) -> None:
    """Save a dict as JSON (non-serialisable values stringified)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def append_experiment_log(row: dict, log_path: str) -> None:
    """Append a row to experiment_log.csv, writing the header if new."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=EXPERIMENT_LOG_COLUMNS, extrasaction="ignore"
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_experiment_log_row(
    metrics_record: dict,
    paths_rel: Dict[str, str],
    timestamp: str,
) -> dict:
    """Flatten a metrics record into a single experiment_log.csv row."""
    ds = metrics_record["dataset"]
    mod = metrics_record["model"]
    met = metrics_record["metrics"]
    tr = metrics_record["training"]
    return {
        "timestamp": timestamp,
        "experiment_id": metrics_record["experiment_id"],
        "dataset": ds["name"],
        "model": mod.get("model_name"),
        "model_type": mod.get("model_type"),
        "input_type": ds["input_type"],
        "input_len": ds["input_len"],
        "horizon": ds["horizon"],
        "seed": ds["seed"],
        "num_features": ds.get("num_features"),
        "total_parameters": mod.get("total_parameters"),
        "trainable_parameters": mod.get("trainable_parameters"),
        "epochs_requested": tr.get("epochs_requested"),
        "epochs_ran": tr.get("epochs_ran"),
        "best_epoch": tr.get("best_epoch"),
        "best_val_loss": tr.get("best_val_loss"),
        "stopped_early": tr.get("stopped_early"),
        "mae": met.get("mae"),
        "rmse": met.get("rmse"),
        "wape": met.get("wape"),
        "checkpoint_path": paths_rel.get("checkpoint", ""),
        "metrics_path": paths_rel["metrics"],
        "predictions_path": paths_rel["predictions"],
        "config_path": paths_rel["config_snapshot"],
    }


def apply_overrides(config: dict, overrides: Optional[dict]) -> dict:
    """Apply CLI overrides into the config so snapshots reflect actual values."""
    if not overrides:
        return config
    t = config["training"]
    if overrides.get("epochs") is not None:
        t["epochs"] = overrides["epochs"]
    if overrides.get("lr") is not None:
        t["learning_rate"] = overrides["lr"]
    if overrides.get("grad_clip") is not None:
        t["grad_clip"] = overrides["grad_clip"]
    if overrides.get("patience") is not None:
        t.setdefault("early_stopping", {})["patience"] = overrides["patience"]
    return config


def run_experiment_from_config(
    config: dict,
    overrides: Optional[dict] = None,
    project_root: str = ".",
    results_dir: Optional[str] = None,
) -> dict:
    """Run one full, recorded experiment from a config dict.

    Steps: apply overrides -> validate -> seed -> data pipeline -> model ->
    train (or skip) -> load best checkpoint -> predict -> metrics -> save
    predictions/metrics/figures/config snapshot -> append experiment_log.csv.
    """
    apply_overrides(config, overrides)
    validate_config(config, project_root=project_root)
    validate_model_config(config)

    if results_dir is None:
        results_dir = os.path.join(project_root, "results")

    experiment_id = build_experiment_id(config)
    paths = build_output_paths(experiment_id, results_dir=results_dir)
    paths_rel = build_output_paths(experiment_id, results_dir="results")

    train_cfg = config["training"]
    seed = train_cfg["seed"]
    set_seed(seed)
    device = get_device()
    print(f"[{experiment_id}] device={device}")

    data = build_data_pipeline(config, project_root=project_root)
    model = build_model(
        config, num_features=data["num_features"], feature_cols=data["feature_cols"]
    ).to(device)
    summary = get_model_summary(model, model_name=config["model"]["name"])

    epochs = train_cfg.get("epochs", 10)
    grad_clip = train_cfg.get("grad_clip")
    es_cfg = train_cfg.get("early_stopping", {})
    criterion = nn.MSELoss()

    is_trainable = count_parameters(model) > 0
    history = {"train_loss": [], "val_loss": [], "learning_rates": []}

    if is_trainable:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        scheduler = build_scheduler(optimizer, config)
        history = fit_model(
            model, data["train_loader"], data["val_loader"], optimizer, criterion,
            device, epochs=epochs, checkpoint_path=paths["checkpoint"], config=config,
            grad_clip=grad_clip, scheduler=scheduler,
            early_stopping_patience=es_cfg.get("patience", 10),
            early_stopping_min_delta=es_cfg.get("min_delta", 0.0),
        )
        if os.path.exists(paths["checkpoint"]):
            load_checkpoint(model, paths["checkpoint"])
            model.to(device)
    else:
        print(f"[{experiment_id}] {config['model']['name']} has no parameters; skip training.")

    y_true_scaled, y_pred_scaled = predict(model, data["test_loader"], device)
    metrics = compute_metrics(y_true_scaled, y_pred_scaled, data["scaler_y"])
    print(f"[{experiment_id}] metrics={metrics}")

    training_summary = {
        "epochs_requested": epochs,
        "epochs_ran": history.get("epochs_ran", len(history["train_loss"])),
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
        "stopped_early": history.get("stopped_early", False),
        "seed": seed,
    }

    checkpoint_rel = paths_rel["checkpoint"] if (is_trainable and os.path.exists(paths["checkpoint"])) else ""
    metrics_record = {
        "experiment_id": experiment_id,
        "dataset": {
            "name": config["dataset"]["name"],
            "input_type": config["dataset"]["input_type"],
            "input_len": config["window"]["input_len"],
            "horizon": config["window"]["horizon"],
            "seed": seed,
            "num_features": data["num_features"],
        },
        "model": summary,
        "metrics": metrics,
        "training": training_summary,
        "paths": {
            "checkpoint": checkpoint_rel,
            "predictions": paths_rel["predictions"],
            "figures": [paths_rel["loss_curve"], paths_rel["prediction_plot"]],
            "config": paths_rel["config_snapshot"],
        },
    }

    # Persist predictions, metrics, history, config snapshot and figures.
    df = create_prediction_dataframe(
        y_true_scaled, y_pred_scaled, data["test_y_dates"], data["scaler_y"]
    )
    os.makedirs(os.path.dirname(paths["predictions"]), exist_ok=True)
    df.to_csv(paths["predictions"], index=False)

    save_json(metrics_record, paths["metrics"])
    save_json(history, paths["history"])
    save_config_snapshot(config, paths["config_snapshot"])
    save_loss_curve(history, paths["loss_curve"], title=experiment_id)
    save_prediction_plot(df, paths["prediction_plot"], title=experiment_id)

    timestamp = datetime.now(timezone.utc).isoformat()
    row = build_experiment_log_row(metrics_record, dict(paths_rel, checkpoint=checkpoint_rel), timestamp)
    append_experiment_log(row, os.path.join(results_dir, "metrics", "experiment_log.csv"))

    print(f"[{experiment_id}] saved metrics, predictions, figures and log row.")
    return metrics_record


def run_experiment(
    config_path: str,
    overrides: Optional[dict] = None,
    project_root: str = ".",
    results_dir: Optional[str] = None,
) -> dict:
    """Load a config file and run one recorded experiment."""
    config = load_config(config_path)
    return run_experiment_from_config(
        config, overrides=overrides, project_root=project_root, results_dir=results_dir
    )
