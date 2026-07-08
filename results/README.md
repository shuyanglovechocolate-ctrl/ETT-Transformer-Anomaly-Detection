# Results Manifest

This directory separates **canonical final artefacts** (committed to git, the
authoritative results referenced by the top-level `README.md` and the thesis) from
**regenerable intermediate artefacts** (git-ignored, recreated by the scripts in
`experiments/`). If you clone this repository, the committed files below are the final
result set; everything ignored can be regenerated with the "How to Reproduce" steps in
the root `README.md`.

## Canonical final artefacts (committed)

### Forecasting (`results/metrics/`)
- `experiment_log.csv` — audit ledger of all 192 forecasting runs (dataset × model ×
  input type × horizon × seed). Source of truth for the forecasting analysis, not a
  transient run log.
- `model_comparison.csv`, `horizon_comparison.csv`, `input_type_comparison.csv` —
  aggregated forecasting comparison tables.
- `per_horizon_summary.csv` — per-horizon metrics aggregated across seeds.
- `best_model_by_dataset_horizon.csv` — best model per (dataset, input type, horizon).
- `model_significance_tests.csv` — paired bootstrap CIs on per-window MAE differences.
- `efficiency_complexity_summary.csv` — accuracy vs parameter count / checkpoint size per
  (dataset, model, input type, horizon). The `mae_per_1k_params` column is transparency
  only, not an efficiency ranking (dividing error by parameters favours larger models).
- `eda_summary_statistics.csv`, `data_pipeline_ETTh1_multivariate_h24.json`,
  `result_validation_report.json` — EDA / pipeline / validation snapshots.
- `reproducibility_manifest.json` — environment snapshot (Python, platform, package
  versions, compute device, git commit); regenerate with `make manifest`.

### Anomaly detection (`results/anomaly/metrics/`)
- `anomaly_detection_results_v3.csv` — **canonical** full fixed-threshold detection
  results (detector × threshold × anomaly type × injection seed). `v1`/`v2` were earlier
  iterations, superseded by `v3` and removed from the working tree (still recoverable
  from git history).
- `anomaly_summary_by_detector.csv`, `anomaly_summary_by_type.csv`,
  `anomaly_summary_by_threshold.csv`, `anomaly_summary_by_horizon.csv` — fixed-threshold
  summaries.
- `anomaly_event_summary_by_detector.csv`, `anomaly_event_summary_by_type.csv` —
  event-wise evaluation.
- `anomaly_threshold_free_results.csv`, `anomaly_threshold_free_summary.csv` — PR-AUC /
  ROC-AUC / best-F1 (threshold-free).
- `anomaly_hybrid_results.csv`, `anomaly_hybrid_summary_by_detector.csv`,
  `anomaly_hybrid_summary_by_type.csv`, `anomaly_hybrid_threshold_free_results.csv` —
  residual + flatness hybrid detectors.
- `anomaly_magnitude_sensitivity.csv` — F1 vs anomaly magnitude.
- `frozen_flatness_diagnostics.csv`, `residual_diagnostics.csv` — diagnostics.
- `accuracy_vs_detection.csv`, `accuracy_detection_correlation.csv` — forecasting
  accuracy vs detection cross-analysis.
- `anomaly_significance_tests.csv` — paired bootstrap + Wilcoxon of the residual detector
  vs causal baselines.
- `anomaly_extended_types_results.csv`, `anomaly_extended_types_summary.csv` — separate
  construct-validity stress test on extended, more fault-like anomaly types (drift /
  noise_burst / stuck_with_jitter). Additive; does not modify the core three-type tables.

### External validity (`results/external_validity/`)
- `ettm_forecasting_comparison.csv`, `ettm_forecasting_verdict.csv` — single-seed
  forecasting check on minute-level ETTm1 / ETTm2 (linear-family vs Transformer). The
  heavy training artefacts live in the git-ignored `results_ettm/` directory; only these
  summary tables are committed.

### Figures
- `results/figures/eda_*.png` — committed EDA figures.
- `results/figures/efficiency_*.png` — committed efficiency/complexity figures.
- `results/anomaly/figures/*.png` — committed representative anomaly figures.

## Regenerable intermediate artefacts (git-ignored)

Recreated by the matrix runners and analysis scripts; not committed:

- `results/checkpoints/` — per-run model checkpoints (`*.pt`).
- `results/predictions/` — per-run prediction CSVs.
- `results/logs/` — per-run config/log/metadata.
- `results/figures/*.png` (per-run forecast/loss plots; EDA figures excepted).
- `results/metrics/*_metrics.json`, `results/metrics/per_horizon_metrics.csv` — per-run
  and intermediate metric snapshots.
- `results/anomaly/residuals/`, `results/anomaly/predictions/`,
  `results/anomaly/multimodel_residuals/` — aggregated residuals / validation predictions.
