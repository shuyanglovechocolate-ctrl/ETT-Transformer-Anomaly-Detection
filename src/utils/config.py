"""YAML configuration loading for reproducible experiments."""

from typing import Any, Dict

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
