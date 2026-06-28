"""Chronological train / validation / test splitting for ETT time series.

Time series data must be split in time order, never shuffled, otherwise
future observations leak into the training set. This module only performs the
split; scaling and windowing happen afterwards (see ``preprocessing.py`` and
``dataset.py``).
"""

from typing import Tuple

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series dataframe into train, validation and test sets
    in chronological order.

    This function does NOT shuffle the data. It is designed for time series
    forecasting, where future observations must not leak into the training set.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by date.
    train_ratio : float
        Proportion of data used for training.
    val_ratio : float
        Proportion of data used for validation.
    test_ratio : float
        Proportion of data used for testing.

    Returns
    -------
    train_df, val_df, test_df : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Chronologically split dataframes.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, but got {ratio_sum}")

    if "date" not in df.columns:
        raise ValueError("Input dataframe must contain a 'date' column.")

    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def get_split_dates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Get date ranges for train, validation and test splits.

    Returns
    -------
    dict
        Date ranges for each split.
    """
    return {
        "train_start": train_df["date"].min(),
        "train_end": train_df["date"].max(),
        "val_start": val_df["date"].min(),
        "val_end": val_df["date"].max(),
        "test_start": test_df["date"].min(),
        "test_end": test_df["date"].max(),
    }


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print a summary of train, validation and test splits."""
    total = len(train_df) + len(val_df) + len(test_df)
    print("Time-based Split Summary")
    print("-" * 40)
    print(
        f"Train: {len(train_df)} rows "
        f"({len(train_df) / total:.2%}) | "
        f"{train_df['date'].min()} -> {train_df['date'].max()}"
    )
    print(
        f"Validation: {len(val_df)} rows "
        f"({len(val_df) / total:.2%}) | "
        f"{val_df['date'].min()} -> {val_df['date'].max()}"
    )
    print(
        f"Test: {len(test_df)} rows "
        f"({len(test_df) / total:.2%}) | "
        f"{test_df['date'].min()} -> {test_df['date'].max()}"
    )
    print("-" * 40)
