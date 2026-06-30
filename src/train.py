"""CLI entry point for a single forecasting experiment (Module 3.1).

Usage:
    python src/train.py --config configs/ETTh1_multivariate_h24.yaml
    python src/train.py --config configs/ETTh1_univariate_h24.yaml --epochs 1

Flow: load config -> validate -> seed -> device -> data pipeline -> model ->
train (or skip for parameter-free models) -> predict test -> original-scale
metrics -> save metrics / predictions / loss curve / prediction plot.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config, validate_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.data.pipeline import build_data_pipeline
from src.models import build_model, validate_model_config, count_parameters, get_model_summary
from src.training import (
    fit_model,
    build_scheduler,
    compute_metrics,
    predict,
    create_prediction_dataframe,
    save_loss_curve,
    save_prediction_plot,
    load_checkpoint,
)


def experiment_name(config: dict) -> str:
    """Build a stable experiment id from the config."""
    d = config["dataset"]
    w = config["window"]
    return (
        f"{d['name']}_{config['model']['name']}_{d['input_type']}"
        f"_len{w['input_len']}_h{w['horizon']}_seed{config['training']['seed']}"
    )


def run(
    config: dict,
    project_root: str,
    epochs_override: int = None,
    lr_override: float = None,
    patience_override: int = None,
    grad_clip_override: float = None,
) -> dict:
    validate_config(config, project_root=project_root)
    validate_model_config(config)

    train_cfg = config["training"]
    seed = train_cfg["seed"]
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    data = build_data_pipeline(config, project_root=project_root)
    model = build_model(
        config,
        num_features=data["num_features"],
        feature_cols=data["feature_cols"],
    ).to(device)

    summary = get_model_summary(model, model_name=config["model"]["name"])
    print("Model summary:", summary)

    name = experiment_name(config)
    results = os.path.join(project_root, "results")
    ckpt_path = os.path.join(results, "checkpoints", f"{name}_best.pt")

    epochs = epochs_override if epochs_override is not None else train_cfg.get("epochs", 10)
    lr = lr_override if lr_override is not None else train_cfg.get("learning_rate", 1e-3)
    grad_clip = grad_clip_override if grad_clip_override is not None else train_cfg.get("grad_clip")
    es_cfg = train_cfg.get("early_stopping", {})
    patience = patience_override if patience_override is not None else es_cfg.get("patience", 10)
    min_delta = es_cfg.get("min_delta", 0.0)

    criterion = nn.MSELoss()
    is_trainable = count_parameters(model) > 0
    history = {"train_loss": [], "val_loss": [], "learning_rates": []}

    if is_trainable:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        scheduler = build_scheduler(optimizer, config)
        history = fit_model(
            model,
            data["train_loader"],
            data["val_loader"],
            optimizer,
            criterion,
            device,
            epochs=epochs,
            checkpoint_path=ckpt_path,
            config=config,
            grad_clip=grad_clip,
            scheduler=scheduler,
            early_stopping_patience=patience,
            early_stopping_min_delta=min_delta,
        )
        # Restore the best (validation) weights before testing.
        if os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path)
            model.to(device)
            print(f"Loaded best checkpoint (epoch {history.get('best_epoch')}).")
    else:
        print(f"{config['model']['name']} has no parameters; skipping training.")

    # Predict on the test set and compute original-scale metrics.
    y_true_scaled, y_pred_scaled = predict(model, data["test_loader"], device)
    metrics = compute_metrics(y_true_scaled, y_pred_scaled, data["scaler_y"])
    print("Test metrics (original OT scale):", metrics)

    training_summary = {
        "epochs_requested": epochs,
        "epochs_ran": history.get("epochs_ran", len(history["train_loss"])),
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
        "stopped_early": history.get("stopped_early", False),
        "seed": seed,
    }

    # Save metrics JSON (metrics + training summary + model summary + config).
    metrics_record = {
        "experiment": name,
        "metrics": metrics,
        "training": training_summary,
        "model_summary": summary,
        "config": config,
    }
    metrics_path = os.path.join(results, "metrics", f"{name}_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2, default=str)

    # Save full training history JSON (gitignored logs/).
    history_path = os.path.join(results, "logs", f"{name}_history.json")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    # Save predictions CSV (long format, original scale, overlap preserved).
    df = create_prediction_dataframe(
        y_true_scaled, y_pred_scaled, data["test_y_dates"], data["scaler_y"]
    )
    pred_path = os.path.join(results, "predictions", f"{name}_predictions.csv")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    df.to_csv(pred_path, index=False)

    # Save figures.
    fig_dir = os.path.join(results, "figures")
    save_loss_curve(history, os.path.join(fig_dir, f"{name}_loss.png"), title=name)
    save_prediction_plot(df, os.path.join(fig_dir, f"{name}_prediction.png"), title=name)

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved predictions -> {pred_path}")
    return metrics_record


def main():
    parser = argparse.ArgumentParser(description="Run one forecasting experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (e.g. 1 for a smoke test).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override the learning rate.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early-stopping patience.")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Override gradient-clipping max norm.")
    args = parser.parse_args()

    config = load_config(args.config)
    run(
        config,
        project_root=str(PROJECT_ROOT),
        epochs_override=args.epochs,
        lr_override=args.lr,
        patience_override=args.patience,
        grad_clip_override=args.grad_clip,
    )


if __name__ == "__main__":
    main()
