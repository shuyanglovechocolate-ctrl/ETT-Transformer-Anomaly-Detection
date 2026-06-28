"""Sliding-window construction, PyTorch Dataset and DataLoader for ETT.

Windows are built independently inside each split (train / val / test). They are
NOT built on the full series before splitting, because that would create windows
crossing split boundaries and leak future information into training. Cross-split
windows are therefore dropped by design; each split yields
``len(split) - input_len - horizon + 1`` samples.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def create_sliding_windows(
    data_x: np.ndarray,
    data_y: np.ndarray,
    input_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one time-series split into supervised learning samples.

    Important:
    Apply this separately to train, validation and test splits. Do NOT apply it
    to the full dataset before splitting, otherwise windows would cross split
    boundaries and cause data leakage.

    Parameters
    ----------
    data_x : np.ndarray
        Scaled input features. Shape: [num_time_steps, num_features]
    data_y : np.ndarray
        Scaled target OT values. Shape: [num_time_steps, 1]
    input_len : int
        Number of historical time steps used as model input.
    horizon : int
        Number of future time steps to predict.

    Returns
    -------
    X : np.ndarray
        Input windows. Shape: [num_samples, input_len, num_features]
    y : np.ndarray
        Forecasting targets. Shape: [num_samples, horizon]
    """
    if len(data_x) != len(data_y):
        raise ValueError(
            f"data_x and data_y must have the same length, "
            f"but got {len(data_x)} and {len(data_y)}."
        )

    if data_y.ndim != 2 or data_y.shape[1] != 1:
        raise ValueError(
            f"data_y must have shape [num_time_steps, 1], but got {data_y.shape}."
        )

    num_time_steps = len(data_x)
    num_features = data_x.shape[1]

    num_samples = num_time_steps - input_len - horizon + 1

    if num_samples <= 0:
        raise ValueError(
            "Not enough time steps to create sliding windows. "
            f"Got num_time_steps={num_time_steps}, "
            f"input_len={input_len}, horizon={horizon}. "
            "Need num_time_steps >= input_len + horizon."
        )

    X = np.zeros((num_samples, input_len, num_features), dtype=np.float32)
    y = np.zeros((num_samples, horizon), dtype=np.float32)

    for i in range(num_samples):
        input_start = i
        input_end = i + input_len

        target_start = input_end
        target_end = input_end + horizon

        X[i] = data_x[input_start:input_end]
        y[i] = data_y[target_start:target_end, 0]

    return X, y


class ETTDataset(Dataset):
    """PyTorch Dataset for ETT forecasting samples."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            Input windows. Shape: [num_samples, input_len, num_features]
        y : np.ndarray
            Forecast targets. Shape: [num_samples, horizon]
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"but got {len(X)} and {len(y)}."
            )

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader from X and y arrays."""
    dataset = ETTDataset(X, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def prepare_windowed_dataloaders(
    scaled_data: Dict[str, object],
    input_len: int,
    horizon: int,
    batch_size: int = 64,
) -> Dict[str, object]:
    """Create sliding windows and DataLoaders for train, val and test splits.

    Assumes:
    1. raw data has already been split into train / val / test;
    2. scaler_x and scaler_y were fitted only on the training split;
    3. train / val / test splits were already transformed;
    4. sliding windows are now created separately within each split.

    Parameters
    ----------
    scaled_data : Dict[str, object]
        Output dictionary from prepare_scaled_splits().
    input_len : int
        Number of historical time steps used as input.
    horizon : int
        Number of future time steps to predict.
    batch_size : int
        Batch size for the DataLoaders.

    Returns
    -------
    Dict[str, object]
        Windowed arrays, DataLoaders and metadata.
    """
    train_X, train_y = create_sliding_windows(
        scaled_data["train_x"],
        scaled_data["train_y"],
        input_len=input_len,
        horizon=horizon,
    )

    val_X, val_y = create_sliding_windows(
        scaled_data["val_x"],
        scaled_data["val_y"],
        input_len=input_len,
        horizon=horizon,
    )

    test_X, test_y = create_sliding_windows(
        scaled_data["test_x"],
        scaled_data["test_y"],
        input_len=input_len,
        horizon=horizon,
    )

    train_loader = create_dataloader(
        train_X, train_y, batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_X, val_y, batch_size=batch_size, shuffle=False
    )
    test_loader = create_dataloader(
        test_X, test_y, batch_size=batch_size, shuffle=False
    )

    return {
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_y": test_y,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_len": input_len,
        "horizon": horizon,
        "batch_size": batch_size,
        "num_features": scaled_data["num_features"],
        "feature_cols": scaled_data["feature_cols"],
        "input_type": scaled_data["input_type"],
        "scaler_x": scaled_data["scaler_x"],
        "scaler_y": scaled_data["scaler_y"],
    }


def print_window_summary(windowed_data: Dict[str, object]) -> None:
    """Print a sliding-window and DataLoader summary."""
    print("Sliding Window Summary")
    print("-" * 40)
    print(f"Input type: {windowed_data['input_type']}")
    print(f"Feature columns: {windowed_data['feature_cols']}")
    print(f"Number of features: {windowed_data['num_features']}")
    print(f"Input length: {windowed_data['input_len']}")
    print(f"Horizon: {windowed_data['horizon']}")
    print(f"Batch size: {windowed_data['batch_size']}")
    print()
    print(f"train_X shape: {windowed_data['train_X'].shape}")
    print(f"train_y shape: {windowed_data['train_y'].shape}")
    print(f"val_X shape: {windowed_data['val_X'].shape}")
    print(f"val_y shape: {windowed_data['val_y'].shape}")
    print(f"test_X shape: {windowed_data['test_X'].shape}")
    print(f"test_y shape: {windowed_data['test_y'].shape}")
    print("-" * 40)
