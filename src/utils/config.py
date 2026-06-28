"""YAML configuration loading and validation for reproducible experiments."""

import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def validate_config(
    config: Dict[str, Any],
    project_root: Optional[str] = None,
) -> None:
    """Validate a pipeline configuration, raising on invalid values.

    Checks:
    - input_type is 'univariate' or 'multivariate';
    - split ratios sum to 1.0;
    - input_len, horizon and batch_size are positive;
    - the dataset path exists (only when project_root is provided).

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    project_root : Optional[str]
        If provided, the dataset path is resolved against it and checked for
        existence. If None, the path-existence check is skipped.
    """
    input_type = config["dataset"]["input_type"]
    if input_type not in ("univariate", "multivariate"):
        raise ValueError(
            "input_type must be 'univariate' or 'multivariate'. "
            f"Got: {input_type}"
        )

    ratio_sum = (
        config["split"]["train_ratio"]
        + config["split"]["val_ratio"]
        + config["split"]["test_ratio"]
    )
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, but got {ratio_sum}.")

    input_len = config["window"]["input_len"]
    horizon = config["window"]["horizon"]
    batch_size = config["training"]["batch_size"]

    if input_len <= 0:
        raise ValueError(f"input_len must be > 0, got {input_len}.")
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")

    if project_root is not None:
        dataset_path = os.path.join(project_root, config["dataset"]["path"])
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")


def print_config(config: Dict[str, Any]) -> None:
    """Print a configuration dictionary in a readable format."""
    print("Configuration")
    print("-" * 40)

    for section, values in config.items():
        print(f"[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"{key}: {value}")
        else:
            print(values)
        print()

    print("-" * 40)
