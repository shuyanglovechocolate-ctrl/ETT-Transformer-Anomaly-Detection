"""Model-agnostic training loop with early stopping and refinements.

The same loop trains every model in the library (Linear, NLinear, DLinear, LSTM,
Transformer). Parameter-free baselines such as Naive are handled by the caller,
which checks ``count_parameters(model) > 0`` before building an optimizer.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.evaluator import evaluate_loss
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> float:
    """Train for one epoch and return the average training loss.

    If ``grad_clip`` is given, gradients are clipped to that max norm before the
    optimizer step (helpful for LSTM / Transformer, harmless for linear models).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Build an optional LR scheduler from config['training']['scheduler'].

    Currently supports ``reduce_on_plateau``. Returns None when not configured.
    """
    sched_cfg = config.get("training", {}).get("scheduler")
    if not sched_cfg:
        return None
    name = sched_cfg.get("name")
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 5),
        )
    raise ValueError(f"Unknown scheduler name: {name}")


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    checkpoint_path: Optional[str] = None,
    config: Optional[dict] = None,
    grad_clip: Optional[float] = None,
    scheduler=None,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
) -> Dict[str, object]:
    """Train a model with early stopping, saving the best (val) checkpoint.

    Returns
    -------
    Dict[str, object]
        History: train_loss, val_loss, learning_rates (per epoch) plus
        best_epoch, best_val_loss, stopped_early and epochs_ran.
    """
    early = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode="min",
    )
    history = {"train_loss": [], "val_loss": [], "learning_rates": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=grad_clip
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(lr)

        improved = early.step(val_loss)
        if improved and checkpoint_path is not None:
            save_checkpoint(
                model, optimizer, epoch, val_loss, checkpoint_path, config=config
            )

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | lr={lr:.2e}"
            + ("  <best>" if improved else "")
        )

        if early.should_stop:
            print(f"Early stopping at epoch {epoch} (best epoch {early.best_epoch}).")
            break

    history["best_epoch"] = early.best_epoch
    history["best_val_loss"] = early.best_score
    history["stopped_early"] = early.should_stop
    history["epochs_ran"] = len(history["train_loss"])
    return history
