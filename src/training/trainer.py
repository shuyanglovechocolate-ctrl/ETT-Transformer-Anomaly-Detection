"""Model-agnostic training loop.

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


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return the average training loss."""
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
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


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
) -> Dict[str, list]:
    """Train a model, tracking train/val loss and saving the best checkpoint.

    Returns
    -------
    Dict[str, list]
        History with "train_loss" and "val_loss" per epoch.
    """
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if checkpoint_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, checkpoint_path, config=config
            )

    return history
