"""Decomposition-based DLinear-style forecaster.

Inspired by DLinear (Zeng et al., AAAI 2023). The input window is split into a
trend component (moving average) and a seasonal component (input - trend); each
is projected to the target horizon by its own linear layer and the two forecasts
are summed.

This is a decomposition-based, DLinear-inspired forecaster, NOT a full
reproduction of the original paper. It uses the flattened multivariate window
(variant A: shared/flattened) and outputs the single-target OT horizon directly.
A channel-independent variant could be added later if required.
"""

import torch
import torch.nn as nn

from src.models.base import BaseForecaster


class MovingAverage(nn.Module):
    """Moving-average smoother along the time axis, length-preserving.

    Edges are padded by replicating the first / last step so the output keeps
    the same sequence length for any kernel size.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}.")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [batch, length, channels]
        pad_front = (self.kernel_size - 1) // 2
        pad_end = self.kernel_size - 1 - pad_front
        front = x[:, :1, :].repeat(1, pad_front, 1)
        end = x[:, -1:, :].repeat(1, pad_end, 1)
        padded = torch.cat([front, x, end], dim=1)        # [B, L + k - 1, C]
        smoothed = self.avg(padded.permute(0, 2, 1))       # [B, C, L]
        return smoothed.permute(0, 2, 1)                   # [B, L, C]


class SeriesDecomposition(nn.Module):
    """Split a series into seasonal and trend components."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearForecaster(BaseForecaster):
    """Decomposition-based linear forecaster (variant A: flattened/shared).

    Parameters
    ----------
    input_len, num_features, horizon : int
        Standard forecaster dimensions.
    kernel_size : int
        Moving-average window for trend extraction (odd recommended, default 25).
    """

    def __init__(
        self,
        input_len: int,
        num_features: int,
        horizon: int,
        kernel_size: int = 25,
    ):
        super().__init__(input_len, num_features, horizon)
        self.kernel_size = kernel_size

        self.decomposition = SeriesDecomposition(kernel_size)
        self.flatten = nn.Flatten()

        flat_dim = input_len * num_features
        self.trend_linear = nn.Linear(flat_dim, horizon)
        self.seasonal_linear = nn.Linear(flat_dim, horizon)

    def forward(self, x):
        # x: [batch, input_len, num_features]
        seasonal, trend = self.decomposition(x)
        seasonal_out = self.seasonal_linear(self.flatten(seasonal))  # [B, horizon]
        trend_out = self.trend_linear(self.flatten(trend))            # [B, horizon]
        return seasonal_out + trend_out
