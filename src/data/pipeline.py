"""Unified Module 1 data-pipeline entry point.

`build_data_pipeline(config)` runs the full leakage-free flow
(load -> chronological split -> train-only scaling -> per-split windowing ->
DataLoaders) and returns a single dictionary that Modules 2 and 3 consume
directly. It also assembles reproducibility metadata and can persist it as JSON.
"""

import json
import os
from typing import Any, Dict, Optional

from src.data.loader import load_ett_dataset
from src.data.splitter import time_based_split
from src.data.preprocessing import prepare_scaled_splits
from src.data.dataset import prepare_windowed_dataloaders
from src.utils.config import validate_config


def build_pipeline_metadata(
    config: Dict[str, Any],
    windowed_data: Dict[str, object],
) -> Dict[str, Any]:
    """Assemble JSON-serialisable metadata describing the prepared pipeline."""
    return {
        "dataset": config["dataset"]["name"],
        "input_type": windowed_data["input_type"],
        "feature_cols": list(windowed_data["feature_cols"]),
        "target_col": config["dataset"]["target"],
        "num_features": int(windowed_data["num_features"]),
        "input_len": int(windowed_data["input_len"]),
        "horizon": int(windowed_data["horizon"]),
        "batch_size": int(windowed_data["batch_size"]),
        "train_samples": int(windowed_data["train_X"].shape[0]),
        "val_samples": int(windowed_data["val_X"].shape[0]),
        "test_samples": int(windowed_data["test_X"].shape[0]),
    }


def save_pipeline_metadata(
    metadata: Dict[str, Any],
    project_root: str = ".",
) -> str:
    """Save pipeline metadata as JSON under results/metrics/.

    Returns
    -------
    str
        Path to the written JSON file.
    """
    out_dir = os.path.join(project_root, "results", "metrics")
    os.makedirs(out_dir, exist_ok=True)

    file_name = (
        f"data_pipeline_{metadata['dataset']}_"
        f"{metadata['input_type']}_h{metadata['horizon']}.json"
    )
    out_path = os.path.join(out_dir, file_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_path


def build_data_pipeline(
    config: Dict[str, Any],
    project_root: str = ".",
    save_metadata: bool = False,
) -> Dict[str, object]:
    """Build the full Module 1 data pipeline from a config dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary (see configs/*.yaml).
    project_root : str
        Base directory used to resolve the dataset path and metadata output.
    save_metadata : bool
        If True, persist pipeline metadata to results/metrics/.

    Returns
    -------
    Dict[str, object]
        The windowed-data dictionary (train/val/test loaders, arrays, scalers,
        num_features, horizon, y_dates, ...) plus a "metadata" entry.
    """
    validate_config(config, project_root=project_root)

    dataset_path = os.path.join(project_root, config["dataset"]["path"])
    df = load_ett_dataset(dataset_path)

    train_df, val_df, test_df = time_based_split(
        df=df,
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
    )

    scaled_data = prepare_scaled_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        input_type=config["dataset"]["input_type"],
        target_col=config["dataset"]["target"],
    )

    windowed_data = prepare_windowed_dataloaders(
        scaled_data=scaled_data,
        input_len=config["window"]["input_len"],
        horizon=config["window"]["horizon"],
        batch_size=config["training"]["batch_size"],
    )

    metadata = build_pipeline_metadata(config, windowed_data)
    windowed_data["metadata"] = metadata

    if save_metadata:
        metadata_path = save_pipeline_metadata(metadata, project_root=project_root)
        windowed_data["metadata_path"] = metadata_path

    return windowed_data
