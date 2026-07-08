"""Tests for the reproducibility manifest (Module 8.1)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.write_reproducibility_manifest import (
    build_manifest, _package_versions, TRACKED_PACKAGES,
)


def test_manifest_has_expected_top_level_keys():
    m = build_manifest()
    for key in ("generated_at", "python_version", "python_executable",
                "platform", "compute_device", "git_commit", "package_versions"):
        assert key in m


def test_manifest_python_version_is_string():
    m = build_manifest()
    assert isinstance(m["python_version"], str)
    assert m["python_version"].count(".") >= 1  # e.g. "3.11.7"


def test_platform_block_has_fields():
    m = build_manifest()
    for field in ("system", "release", "machine"):
        assert field in m["platform"]


def test_package_versions_cover_tracked_packages():
    versions = _package_versions()
    assert set(versions) == set(TRACKED_PACKAGES)
    # numpy and pandas are hard dependencies and must resolve to a version string
    assert isinstance(versions["numpy"], str)
    assert isinstance(versions["pandas"], str)


def test_missing_package_reports_none():
    versions = _package_versions(["definitely-not-a-real-package-xyz"])
    assert versions["definitely-not-a-real-package-xyz"] is None
