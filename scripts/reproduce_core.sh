#!/usr/bin/env bash
# Full end-to-end reproduction of the core study (Module 8.1).
#
# This is the HEAVY path: it trains the forecasting matrices from scratch and then
# runs every analysis. On a laptop this takes a long time (the deep models and the
# minute-level ETTm runs dominate). For a quick review, prefer `make test` and
# `make manifest`, which need no training.
#
# Heavy per-run artefacts (checkpoints, predictions, logs) are git-ignored and are
# regenerated here; only summary tables and selected figures are committed.
#
# Override the interpreter with PYTHON=..., e.g. PYTHON=python3.11 ./scripts/reproduce_core.sh
set -euo pipefail

PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

echo "== 0. Environment manifest =="
"$PYTHON" experiments/write_reproducibility_manifest.py

echo "== 1. Module 1 pipeline sanity check =="
"$PYTHON" experiments/check_module1_pipeline.py

echo "== 2. Forecasting experiments (ETTh1/ETTh2) =="
"$PYTHON" experiments/run_matrix.py --matrix core-light --skip-existing
"$PYTHON" experiments/run_matrix.py --matrix core-deep  --skip-existing
"$PYTHON" experiments/run_matrix.py --matrix robustness-deep --skip-existing
"$PYTHON" experiments/summarize_results.py
"$PYTHON" experiments/analyze_forecasting_results.py
"$PYTHON" experiments/analyze_efficiency_complexity.py

echo "== 2b. External-validity check (minute-level ETTm, isolated dir) =="
"$PYTHON" experiments/run_matrix.py --matrix core-light \
  --models naive linear nlinear dlinear transformer \
  --datasets ETTm1 ETTm2 --input-types multivariate \
  --horizons 96 --seeds 42 --results-dir results_ettm --skip-existing
"$PYTHON" experiments/analyze_ettm_external_validity.py

echo "== 3. Anomaly detection (Module 4) =="
"$PYTHON" experiments/prepare_anomaly_residuals.py
"$PYTHON" experiments/run_anomaly_detection.py
"$PYTHON" experiments/summarize_anomaly_results.py
"$PYTHON" experiments/run_magnitude_sensitivity.py
"$PYTHON" experiments/diagnose_anomaly_residuals.py

echo "== 4. Deeper anomaly analysis =="
"$PYTHON" experiments/run_threshold_free_eval.py
"$PYTHON" experiments/run_hybrid_detection.py
"$PYTHON" experiments/prepare_multimodel_residuals.py
"$PYTHON" experiments/analyze_accuracy_vs_detection.py
"$PYTHON" experiments/analyze_anomaly_significance.py
"$PYTHON" experiments/run_extended_anomaly_types.py

echo "== Done. Summary tables are under results/ and results/anomaly/metrics/. =="
