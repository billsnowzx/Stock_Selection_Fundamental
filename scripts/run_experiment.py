from __future__ import annotations

import argparse

from stock_selection_fundamental.research.experiments import run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment suite.")
    parser.add_argument("--config", default="configs/experiments/fundamental_grid.yaml")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None)
    parser.add_argument("--fail-fast", dest="fail_fast", action="store_true")
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.set_defaults(fail_fast=None)
    args = parser.parse_args()
    result = run_experiment_suite(args.config, resume=args.resume, fail_fast=args.fail_fast)
    print(f"Experiment ID: {result.experiment_id}")
    print(f"Output Dir: {result.output_dir.resolve()}")
    print(result.summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
