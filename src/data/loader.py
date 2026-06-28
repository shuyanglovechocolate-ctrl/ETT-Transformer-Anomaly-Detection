"""ETT data loading and quality-check utilities.

This module is part of Module 1 and provides reusable functions to:
- load an ETT dataset from a CSV file,
- parse and sort by timestamp,
- check missing values, duplicated timestamps and time continuity,
- produce a compact data-quality report.

EDA visualisation lives in ``notebooks/01_eda.ipynb`` and calls these helpers.
"""

import pandas as pd


ETT_COLUMNS = [
    "date",
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
    "OT",
]

# Numeric feature columns (everything except the timestamp).
FEATURE_COLUMNS = [
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
    "OT",
]

# Forecasting target.
TARGET_COLUMN = "OT"


def load_ett_dataset(file_path: str) -> pd.DataFrame:
    """Load an ETT dataset from a CSV file.

    This function:
    - reads the CSV file,
    - checks required columns,
    - converts the date column to datetime,
    - sorts the data by time,
    - resets the index.

    Parameters
    ----------
    file_path : str
        Path to the ETT CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned and time-sorted ETT dataframe.
    """
    df = pd.read_csv(file_path)

    missing_columns = [col for col in ETT_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def get_basic_info(df: pd.DataFrame, dataset_name: str = "ETT") -> dict:
    """Get basic information of an ETT dataset.

    Parameters
    ----------
    df : pd.DataFrame
        ETT dataframe.
    dataset_name : str
        Name of the dataset, e.g. ETTh1.

    Returns
    -------
    dict
        Basic dataset information.
    """
    info = {
        "dataset": dataset_name,
        "rows": len(df),
        "columns": list(df.columns),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "num_features": len(df.columns) - 1,
    }

    return info


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Check missing values in each column.

    Parameters
    ----------
    df : pd.DataFrame
        ETT dataframe.

    Returns
    -------
    pd.Series
        Number of missing values per column.
    """
    return df.isna().sum()


def check_duplicate_timestamps(df: pd.DataFrame) -> int:
    """Check duplicated timestamps in the date column.

    Parameters
    ----------
    df : pd.DataFrame
        ETT dataframe.

    Returns
    -------
    int
        Number of duplicated timestamps.
    """
    return int(df["date"].duplicated().sum())


def check_time_interval(df: pd.DataFrame) -> pd.Series:
    """Check time intervals between consecutive observations.

    Parameters
    ----------
    df : pd.DataFrame
        ETT dataframe.

    Returns
    -------
    pd.Series
        Counts of different time intervals.
    """
    return df["date"].diff().value_counts()


def check_time_continuity(
    df: pd.DataFrame,
    expected_freq: str,
) -> dict:
    """Check whether the dataset has missing timestamps.

    Builds the full expected date range from the first to the last timestamp
    at ``expected_freq`` and compares it against the observed timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        ETT dataframe.
    expected_freq : str
        Expected frequency, e.g. "1h" for hourly data, "15min" for minute-level data.

    Returns
    -------
    dict
        Time continuity check results.
    """
    full_range = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq=expected_freq,
    )

    actual_dates = pd.DatetimeIndex(df["date"])
    missing_dates = full_range.difference(actual_dates)

    result = {
        "expected_frequency": expected_freq,
        "expected_num_timestamps": len(full_range),
        "actual_num_timestamps": len(actual_dates),
        "missing_num_timestamps": len(missing_dates),
        "missing_timestamps": missing_dates,
    }

    return result


def run_data_quality_report(
    df: pd.DataFrame,
    dataset_name: str,
    expected_freq: str,
) -> dict:
    """Run a complete data-quality report for an ETT dataset.

    Parameters
    ----------
    df : pd.DataFrame
        ETT dataframe.
    dataset_name : str
        Dataset name.
    expected_freq : str
        Expected time frequency.

    Returns
    -------
    dict
        A full data-quality report.
    """
    report = {
        "basic_info": get_basic_info(df, dataset_name),
        "missing_values": check_missing_values(df),
        "duplicated_timestamps": check_duplicate_timestamps(df),
        "time_interval_counts": check_time_interval(df),
        "time_continuity": check_time_continuity(df, expected_freq),
    }

    return report
