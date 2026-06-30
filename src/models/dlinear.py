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
    """Decomposition-based linear forecaster.

    Two channel-handling modes:

    - ``channel_independent=False`` (default, variant A): trend and seasonal
      components are flattened across all channels and each mapped to the
      horizon by a shared linear layer (channel-mixing).
    - ``channel_independent=True``: each channel is projected over time by its
      own (independent) trend/seasonal weights, then a linear head mixes the
      per-channel forecasts into the target horizon.

    Parameters
    ----------
    input_len, num_features, horizon : int
        Standard forecaster dimensions.
    kernel_size : int
        Moving-average window for trend extraction (odd recommended, default 25).
    channel_independent : bool
        Channel handling mode (see above).
    """

    model_type = "linear"
    description = "Decomposition-based DLinear-inspired forecaster (trend + seasonal)."

    def __init__(
        self,
        input_len: int,
        num_features: int,
        horizon: int,
        kernel_size: int = 25,
        channel_independent: bool = False,
    ):
        super().__init__(input_len, num_features, horizon)
        self.kernel_size = kernel_size
        self.channel_independent = channel_independent

        self.decomposition = SeriesDecomposition(kernel_size)

        if channel_independent:
            # Independent temporal projection per channel: weight [C, L, horizon].
            self.seasonal_weight = nn.Parameter(
                torch.empty(num_features, input_len, horizon)
            )
            self.trend_weight = nn.Parameter(
                torch.empty(num_features, input_len, horizon)
            )
            self.seasonal_bias = nn.Parameter(torch.zeros(num_features, horizon))
            self.trend_bias = nn.Parameter(torch.zeros(num_features, horizon))
            nn.init.xavier_uniform_(self.seasonal_weight)
            nn.init.xavier_uniform_(self.trend_weight)
            # Mix per-channel forecasts into the single target horizon.
            self.mix = nn.Linear(num_features * horizon, horizon)
        else:
            self.flatten = nn.Flatten()
            flat_dim = input_len * num_features
            self.trend_linear = nn.Linear(flat_dim, horizon)
            self.seasonal_linear = nn.Linear(flat_dim, horizon)

    def forward(self, x):
        # x: [batch, input_len, num_features]
        seasonal, trend = self.decomposition(x)

        if self.channel_independent:
            s = seasonal.permute(0, 2, 1)   # [B, C, L]
            t = trend.permute(0, 2, 1)
            s_out = torch.einsum("bcl,clh->bch", s, self.seasonal_weight)
            s_out = s_out + self.seasonal_bias
            t_out = torch.einsum("bcl,clh->bch", t, self.trend_weight)
            t_out = t_out + self.trend_bias
            combined = s_out + t_out         # [B, C, horizon]
            return self.mix(combined.flatten(1))  # [B, horizon]

        seasonal_out = self.seasonal_linear(self.flatten(seasonal))  # [B, horizon]
        trend_out = self.trend_linear(self.flatten(trend))            # [B, horizon]
        return seasonal_out + trend_out
