"""Write a reproducibility manifest (Module 8.1).

Records the environment a result set was produced in — Python, platform, key package
versions, the selected compute device and the current git commit — so a marker can see
exactly what was used and compare it against their own environment. The committed
manifest is a snapshot of the author's environment; re-running this script on another
machine regenerates it with that machine's versions.

Writes results/metrics/reproducibility_manifest.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Distribution names (as installed), not import names.
TRACKED_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
                    "torch", "PyYAML", "tqdm", "pytest"]


def _package_versions(packages=TRACKED_PACKAGES) -> dict:
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def _git_commit(project_root) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _device() -> str:
    try:
        from src.utils.device import get_device
        return str(get_device())
    except Exception:  # torch missing or backend error — manifest must not crash
        return None


def build_manifest(project_root=PROJECT_ROOT) -> dict:
    """Assemble the reproducibility manifest as a plain dict."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "compute_device": _device(),
        "git_commit": _git_commit(project_root),
        "package_versions": _package_versions(),
    }


def main():
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest.")
    parser.add_argument("--out", default=str(
        PROJECT_ROOT / "results" / "metrics" / "reproducibility_manifest.json"))
    args = parser.parse_args()

    manifest = build_manifest()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {args.out}")
    print(f"  python={manifest['python_version']} device={manifest['compute_device']} "
          f"torch={manifest['package_versions'].get('torch')}")


if __name__ == "__main__":
    main()
