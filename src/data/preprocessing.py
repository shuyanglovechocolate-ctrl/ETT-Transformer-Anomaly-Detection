"""Leakage-free scaling and feature selection for ETT forecasting.

Critical processing order enforced by this module:

    raw data
    -> time-based split (see splitter.py)
    -> fit scaler on train only
    -> transform train / val / test with the same scalers
    -> create sliding windows inside each split (see dataset.py)

Two scalers are maintained:
- ``scaler_x`` standardises the input features (1 column for univariate,
  7 columns for multivariate).
- ``scaler_y`` is fitted only on the training OT column and is used to
  inverse-transform model predictions and ground truth back to the original
  oil-temperature scale, so MAE / RMSE and anomaly residuals are computed in
  interpretable units.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


MULTIVARIATE_FEATURES = [
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
    "OT",
]

UNIVARIATE_FEATURES = ["OT"]

TARGET_COLUMN = "OT"


def get_feature_columns(input_type: str) -> List[str]:
    """Get input feature columns based on input type.

    Parameters
    ----------
    input_type : str
        Either 'univariate' or 'multivariate'.

    Returns
    -------
    List[str]
        Selected feature columns.
    """
    if input_type == "univariate":
        return UNIVARIATE_FEATURES
    if input_type == "multivariate":
        return MULTIVARIATE_FEATURES
    raise ValueError(
        "input_type must be either 'univariate' or 'multivariate'. "
        f"Got: {input_type}"
    )


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_COLUMN,
) -> Tuple[StandardScaler, StandardScaler]:
    """Fit scaler_x and scaler_y using training data only.

    scaler_x:
        Fitted on the selected input features.
    scaler_y:
        Fitted only on the target OT column. Used for inverse-transforming
        model outputs back to the original scale.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    feature_cols : List[str]
        Input feature columns.
    target_col : str
        Target column, default is 'OT'.

    Returns
    -------
    scaler_x, scaler_y : Tuple[StandardScaler, StandardScaler]
        Feature scaler and target scaler.
    """
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaler_x.fit(train_df[feature_cols])
    scaler_y.fit(train_df[[target_col]])

    return scaler_x, scaler_y


def transform_split(
    df: pd.DataFrame,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    feature_cols: List[str],
    target_col: str = TARGET_COLUMN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform one split using training-fitted scalers.

    Parameters
    ----------
    df : pd.DataFrame
        Train, validation or test dataframe.
    scaler_x : StandardScaler
        Feature scaler fitted on training features.
    scaler_y : StandardScaler
        Target scaler fitted on training OT only.
    feature_cols : List[str]
        Input feature columns.
    target_col : str
        Target column.

    Returns
    -------
    x_scaled : np.ndarray
        Scaled input features. Shape: [num_time_steps, num_features]
    y_scaled : np.ndarray
        Scaled target values. Shape: [num_time_steps, 1]
    """
    x_scaled = scaler_x.transform(df[feature_cols])
    y_scaled = scaler_y.transform(df[[target_col]])
    return x_scaled, y_scaled


def prepare_scaled_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_type: str,
    target_col: str = TARGET_COLUMN,
) -> Dict[str, object]:
    """Prepare scaled train, validation and test splits.

    Critical processing order:
    1. Split the raw time series into train / val / test (done beforehand).
    2. Fit scaler_x and scaler_y on training data only.
    3. Transform train / val / test using the same training-fitted scalers.
    4. Sliding windows are created separately after this step.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Time-based data splits.
    input_type : str
        Either 'univariate' or 'multivariate'.
    target_col : str
        Target column.

    Returns
    -------
    Dict[str, object]
        Dictionary containing scaled arrays, scalers and metadata.
    """
    feature_cols = get_feature_columns(input_type)

    scaler_x, scaler_y = fit_scalers(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    train_x, train_y = transform_split(
        train_df, scaler_x, scaler_y, feature_cols, target_col
    )
    val_x, val_y = transform_split(
        val_df, scaler_x, scaler_y, feature_cols, target_col
    )
    test_x, test_y = transform_split(
        test_df, scaler_x, scaler_y, feature_cols, target_col
    )

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "num_features": len(feature_cols),
        "input_type": input_type,
    }


def inverse_transform_y(
    y_scaled: np.ndarray,
    scaler_y: StandardScaler,
) -> np.ndarray:
    """Inverse-transform scaled target values back to the original OT scale.

    This function uses scaler_y, not scaler_x.

    Parameters
    ----------
    y_scaled : np.ndarray
        Scaled target values. Can be shape [N], [N, 1] or [N, horizon].
    scaler_y : StandardScaler
        Target scaler fitted only on training OT.

    Returns
    -------
    np.ndarray
        Target values in the original OT scale, same shape as the input.
    """
    original_shape = y_scaled.shape
    y_reshaped = np.asarray(y_scaled).reshape(-1, 1)
    y_original = scaler_y.inverse_transform(y_reshaped)
    return y_original.reshape(original_shape)


def print_preprocessing_summary(scaled_data: Dict[str, object]) -> None:
    """Print a summary of preprocessing outputs."""
    print("Preprocessing Summary")
    print("-" * 40)
    print(f"Input type: {scaled_data['input_type']}")
    print(f"Feature columns: {scaled_data['feature_cols']}")
    print(f"Target column: {scaled_data['target_col']}")
    print(f"Number of features: {scaled_data['num_features']}")
    print(f"train_x shape: {scaled_data['train_x'].shape}")
    print(f"train_y shape: {scaled_data['train_y'].shape}")
    print(f"val_x shape: {scaled_data['val_x'].shape}")
    print(f"val_y shape: {scaled_data['val_y'].shape}")
    print(f"test_x shape: {scaled_data['test_x'].shape}")
    print(f"test_y shape: {scaled_data['test_y'].shape}")
    print("-" * 40)
