"""Checkpoint saving and loading for trainable models."""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    path: str,
    config: Optional[dict] = None,
) -> None:
    """Save a checkpoint with model/optimizer state and minimal metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "config": config,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """Load a checkpoint into model (and optimizer if given). Returns the dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
