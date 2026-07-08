"""Input-length sensitivity ablation runner (Module 8.3).

Trains a small forecasting grid that varies only the input window length, to test
whether the core model ranking depends on the default ``input_len=96``. Everything is
written to an ISOLATED results directory (default ``results_inputlen/``, git-ignored),
so the canonical ETTh study backing the v1.0-thesis release is never touched.

Grid: ETTh1 x multivariate x horizon 96 x seed 42 x
      input_len in {48, 96, 192} x models {nlinear, dlinear, transformer}
= 9 runs. Reuses the tested run_matrix build/run helpers (no retraining logic here).

Run:  python experiments/run_input_len_ablation.py
Then: python experiments/analyze_input_len_ablation.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.run_matrix import build_matrix, run_matrix

ABLATION_INPUT_LENS = [48, 96, 192]
ABLATION_MODELS = ["nlinear", "dlinear", "transformer"]
ABLATION_DATASET = "ETTh1"
ABLATION_HORIZON = 96
ABLATION_SEED = 42


def build_ablation_configs(input_lens=ABLATION_INPUT_LENS, models=ABLATION_MODELS):
    configs = []
    for input_len in input_lens:
        configs.extend(build_matrix(
            models=models, datasets=[ABLATION_DATASET],
            input_types=["multivariate"], horizons=[ABLATION_HORIZON],
            seeds=[ABLATION_SEED], input_len=input_len))
    return configs


def main():
    parser = argparse.ArgumentParser(description="Input-length sensitivity ablation.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results_inputlen"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    configs = build_ablation_configs()
    print(f"Input-length ablation: {len(configs)} experiment(s).")
    run_matrix(configs, project_root=str(PROJECT_ROOT),
               results_dir=args.results_dir, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
