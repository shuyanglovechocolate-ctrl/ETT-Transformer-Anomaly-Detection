"""Tests for the pilot experiment matrix runner (Module 3.4a)."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.run_matrix import (
    build_pilot_matrix,
    build_matrix_config,
    get_default_model_config,
    experiment_exists,
    run_matrix,
    PILOT_MODELS,
)
from src.models import validate_model_config
from src.training.experiment import build_experiment_id, build_output_paths


def test_build_pilot_matrix_default_has_six():
    configs = build_pilot_matrix()
    assert len(configs) == 6
    names = [c["model"]["name"] for c in configs]
    assert names == PILOT_MODELS


def test_build_pilot_matrix_model_subset():
    configs = build_pilot_matrix(models=["naive", "linear"])
    assert [c["model"]["name"] for c in configs] == ["naive", "linear"]


def test_matrix_configs_have_full_model_section_and_validate():
    for config in build_pilot_matrix():
        assert "model" in config and "name" in config["model"]
        # Each generated config must pass model-config validation.
        validate_model_config(config)


def test_default_model_config_has_hyperparams():
    lstm = get_default_model_config("lstm")
    assert lstm["hidden_dim"] == 64 and lstm["num_layers"] == 2
    tr = get_default_model_config("transformer")
    assert tr["d_model"] == 64 and tr["nhead"] == 4
    # Returns a fresh dict each call (no shared mutable state).
    lstm["hidden_dim"] = 999
    assert get_default_model_config("lstm")["hidden_dim"] == 64


def test_get_default_model_config_unknown_raises():
    with pytest.raises(ValueError):
        get_default_model_config("bogus")


def test_experiment_exists_detects_metrics(tmp_path):
    config = build_matrix_config("ETTh1", "multivariate", 96, 24, 42, "linear")
    assert experiment_exists(config, str(tmp_path)) is False
    # Create the metrics file the runner would check for.
    eid = build_experiment_id(config)
    metrics_path = Path(build_output_paths(eid, str(tmp_path))["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}")
    assert experiment_exists(config, str(tmp_path)) is True


def test_run_matrix_skip_and_force_mutually_exclusive():
    with pytest.raises(ValueError):
        run_matrix([], skip_existing=True, force=True)


def test_run_matrix_skips_existing(tmp_path):
    config = build_matrix_config("ETTh1", "multivariate", 96, 24, 42, "naive")
    eid = build_experiment_id(config)
    metrics_path = Path(build_output_paths(eid, str(tmp_path))["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}")

    summary = run_matrix([config], project_root=str(PROJECT_ROOT),
                         results_dir=str(tmp_path), skip_existing=True)
    assert summary["skipped"] == [eid]
    assert summary["succeeded"] == []


def test_run_matrix_fail_soft(tmp_path):
    # One invalid config (bogus model) must not stop a valid one; both outcomes
    # are recorded. Naive runs fast (no training).
    good = build_matrix_config("ETTh1", "multivariate", 96, 24, 42, "naive")
    bad = build_matrix_config("ETTh1", "multivariate", 96, 24, 42, "naive")
    bad["model"] = {"name": "bogus"}

    summary = run_matrix([bad, good], project_root=str(PROJECT_ROOT),
                         results_dir=str(tmp_path))
    assert len(summary["failed"]) == 1
    assert len(summary["succeeded"]) == 1
    assert summary["failed"][0]["model"] == "bogus"
