export type ComparisonRow = {
  dataset: string;
  model: string;
  input_type: "univariate" | "multivariate";
  horizon: number;
  mae: number;
  mae_std: number;
  rmse: number;
  rmse_std: number;
  wape: number;
  wape_std: number;
  num_runs: number;
};

export type PredIndexEntry = { key: string; dataset: string; model: string };

export type PredSeries = {
  dataset: string;
  model: string;
  horizon: number;
  dates: string[];
  y_true: (number | null)[];
  y_pred: (number | null)[];
};

export type AnomalyData = {
  by_type: {
    anomaly_type: string;
    best_detector: string;
    best_mean_f1: number;
    mean_recall: number;
    mean_precision: number;
  }[];
  threshold_free: {
    detector_type: string;
    anomaly_type: string;
    pr_auc: number;
    pr_auc_std: number;
    roc_auc: number;
    best_f1: number;
    num_runs: number;
  }[];
  accuracy_vs_detection: {
    dataset: string;
    model: string;
    horizon: number;
    anomaly_type: string;
    forecast_mae: number;
    average_precision: number;
    event_recall: number;
  }[];
  magnitude_sensitivity: {
    anomaly_type: string;
    magnitude_scale: number;
    f1: number;
  }[];
};

export type EfficiencyRow = {
  dataset: string;
  model: string;
  horizon: number;
  input_type: "univariate" | "multivariate";
  mae: number;
  mae_std: number;
  params: number;
  checkpoint_mb: number | null;
  epochs: number | null;
};

export type FrozenData = {
  detectors: {
    name: string;
    label: string;
    f1: number;
    recall: number;
    precision: number;
    event_recall: number;
  }[];
  diagnosis: { flatness_ratio: number; residual_ratio: number };
  contrast: Record<string, number>;
};

export type Manifest = {
  headline: {
    best_mae: number;
    num_models: number;
    num_datasets: number;
    num_horizons: number;
    num_seeds: number;
    num_anomaly_types: number;
  };
  reproducibility: {
    compute_device: string;
    generated_at: string;
    git_commit: string;
    package_versions: Record<string, string>;
    python_version: string;
    platform: Record<string, string>;
  };
};
