"""Base class for all forecasting models.

Every model in the library follows the same input-output contract so that
Module 3's training loop never needs to know the internals:

    x      : [batch_size, input_len, num_features]
    y_pred : [batch_size, horizon]
"""

import torch.nn as nn


class BaseForecaster(nn.Module):
    """Common interface for forecasting models.

    Parameters
    ----------
    input_len : int
        Number of historical time steps in each input window.
    num_features : int
        Number of input features (1 for univariate, 7 for multivariate).
    horizon : int
        Number of future OT steps to predict.
    """

    def __init__(self, input_len: int, num_features: int, horizon: int):
        super().__init__()
        self.input_len = input_len
        self.num_features = num_features
        self.horizon = horizon

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward().")
