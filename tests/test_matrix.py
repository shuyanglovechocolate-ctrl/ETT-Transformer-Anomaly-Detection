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
    build_core_light_matrix,
    build_core_deep_matrix,
    build_robustness_deep_matrix,
    build_matrix_configs,
    get_default_model_config,
    experiment_exists,
    is_completed_metrics_file,
    run_matrix,
    REQUIRED_METRICS_KEYS,
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


def _complete_metrics_json():
    return {k: "x" for k in REQUIRED_METRICS_KEYS}


def test_run_matrix_skips_existing(tmp_path):
    config = build_matrix_config("ETTh1", "multivariate", 96, 24, 42, "naive")
    eid = build_experiment_id(config)
    metrics_path = Path(build_output_paths(eid, str(tmp_path))["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_complete_metrics_json()))

    summary = run_matrix([config], project_root=str(PROJECT_ROOT),
                         results_dir=str(tmp_path), skip_existing=True)
    assert summary["skipped"] == [eid]
    assert summary["succeeded"] == []


def test_is_completed_metrics_file(tmp_path):
    missing = tmp_path / "missing.json"
    assert is_completed_metrics_file(str(missing)) is False

    empty = tmp_path / "empty.json"
    empty.write_text("")
    assert is_completed_metrics_file(str(empty)) is False

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    assert is_completed_metrics_file(str(bad)) is False

    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"experiment_id": "x"}))  # missing keys
    assert is_completed_metrics_file(str(partial)) is False

    good = tmp_path / "good.json"
    good.write_text(json.dumps(_complete_metrics_json()))
    assert is_completed_metrics_file(str(good)) is True


def test_run_matrix_does_not_skip_broken_metrics(tmp_path):
    # A half-written / empty metrics file must NOT be treated as complete: the
    # experiment should re-run rather than be skipped.
    config = build_matrix_config("ETTh1", "multivariate", 96, 24, 42, "naive")
    eid = build_experiment_id(config)
    metrics_path = Path(build_output_paths(eid, str(tmp_path))["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("")  # broken (empty) file

    summary = run_matrix([config], project_root=str(PROJECT_ROOT),
                         results_dir=str(tmp_path), skip_existing=True)
    assert summary["skipped"] == []
    assert summary["succeeded"] == [eid]


def test_build_core_light_default_144():
    configs = build_core_light_matrix()
    assert len(configs) == 144  # 4 models x 2 datasets x 2 input_types x 3 horizons x 3 seeds


def test_build_core_deep_default_36_multivariate_only():
    configs = build_core_deep_matrix()
    assert len(configs) == 36  # 2 x 2 datasets x 1 input_type x 3 horizons x 3 seeds
    assert all(c["dataset"]["input_type"] == "multivariate" for c in configs)
    assert set(c["model"]["name"] for c in configs) == {"lstm", "transformer"}


def test_core_light_filters():
    configs = build_core_light_matrix(
        datasets=["ETTh1"], input_types=["multivariate"],
        horizons=[24], seeds=[42], models=["naive", "linear"],
    )
    assert len(configs) == 2
    assert all(c["dataset"]["name"] == "ETTh1" for c in configs)
    assert all(c["window"]["horizon"] == 24 for c in configs)


def test_build_matrix_configs_dispatch_and_validate():
    for matrix, expected in [("pilot", 6), ("core-light", 144),
                             ("core-deep", 36), ("robustness-deep", 12)]:
        configs = build_matrix_configs(matrix)
        assert len(configs) == expected
    # A sample of core configs must pass model validation.
    for config in build_core_light_matrix(datasets=["ETTh2"], horizons=[48], seeds=[42]):
        validate_model_config(config)


def test_robustness_deep_matrix_regularized():
    configs = build_robustness_deep_matrix()
    assert len(configs) == 12  # 2 datasets x 1 horizon x 3 seeds x 2 models
    for c in configs:
        assert c["window"]["horizon"] == 96
        assert c["dataset"]["input_type"] == "multivariate"
        assert c["experiment"]["tag"] == "regularized"
        assert c["training"]["weight_decay"] == 1e-4
        validate_model_config(c)
    # Regularization actually applied.
    tr = [c for c in configs if c["model"]["name"] == "transformer"][0]
    assert tr["training"]["learning_rate"] == 1e-4
    assert tr["model"]["dropout"] == 0.2
    lstm = [c for c in configs if c["model"]["name"] == "lstm"][0]
    assert lstm["model"]["dropout"] == 0.3


def test_experiment_id_includes_regularized_tag():
    config = build_robustness_deep_matrix(
        datasets=["ETTh1"], seeds=[42], models=["transformer"])[0]
    assert build_experiment_id(config) == \
        "ETTh1_transformer_regularized_multivariate_len96_h96_seed42"


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
