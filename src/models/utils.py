"""Model utilities: parameter counting and summaries.

Used by Module 3 to record model complexity alongside experiment results.
"""

from typing import Dict, Optional

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Parameters
    ----------
    model : nn.Module
        Any model.
    trainable_only : bool
        If True, count only parameters with requires_grad=True.

    Returns
    -------
    int
        Number of parameters (0 for parameter-free models such as Naive).
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_summary(
    model: nn.Module,
    model_name: Optional[str] = None,
) -> Dict[str, object]:
    """Build a serialisable summary of a model for experiment logging.

    Parameters
    ----------
    model : nn.Module
        The model to summarise.
    model_name : Optional[str]
        Optional config name (e.g. "dlinear"). Defaults to the class name.

    Returns
    -------
    Dict[str, object]
        Summary including parameter counts and, when available, the standard
        forecaster dimensions (input_len, num_features, horizon).
    """
    summary = {
        "model_name": model_name if model_name is not None else type(model).__name__,
        "class_name": type(model).__name__,
        "total_parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model, trainable_only=True),
    }

    for attr in ("input_len", "num_features", "horizon"):
        if hasattr(model, attr):
            summary[attr] = getattr(model, attr)

    return summary
