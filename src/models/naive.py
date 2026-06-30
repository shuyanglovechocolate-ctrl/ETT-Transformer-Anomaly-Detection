"""Naive / persistence baseline.

Predicts the last observed OT value for every future step. Requires no training
and serves as a sanity check: a deep model that cannot beat persistence is not
learning anything useful.
"""

from typing import List

import torch

from src.models.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Repeat the last input OT value across the whole horizon.

    The OT column index is resolved from ``feature_cols`` rather than hard-coded,
    so the model works for both univariate (OT at index 0) and multivariate
    (OT typically the last column) inputs.

    Parameters
    ----------
    input_len, num_features, horizon : int
        Standard forecaster dimensions.
    feature_cols : List[str]
        Ordered input feature names; must contain "OT".
    target_col : str
        Name of the target column, default "OT".
    """

    def __init__(
        self,
        input_len: int,
        num_features: int,
        horizon: int,
        feature_cols: List[str],
        target_col: str = "OT",
    ):
        super().__init__(input_len, num_features, horizon)
        if target_col not in feature_cols:
            raise ValueError(
                f"target_col '{target_col}' not found in feature_cols {feature_cols}."
            )
        self.ot_index = feature_cols.index(target_col)

    def forward(self, x):
        # x: [batch, input_len, num_features]
        last_ot = x[:, -1, self.ot_index]          # [batch]
        return last_ot.unsqueeze(1).repeat(1, self.horizon)  # [batch, horizon]
