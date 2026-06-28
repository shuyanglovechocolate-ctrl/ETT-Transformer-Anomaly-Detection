"""Reproducibility utilities: global random seed."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CUDA behaviour more deterministic. This may slightly reduce training
    # speed but improves reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
