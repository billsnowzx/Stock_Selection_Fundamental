from __future__ import annotations

import argparse

from stock_selection_fundamental.research.experiments import run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment suite.")
    parser.add_argument("--config", default="configs/experiments/fundamental_grid.yaml")
    args = parser.parse_args()
    result = run_experiment_suite(args.config)
    print(f"Experiment ID: {result.experiment_id}")
    print(f"Output Dir: {result.output_dir.resolve()}")
    print(result.summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
