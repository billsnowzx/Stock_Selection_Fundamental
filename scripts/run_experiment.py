from __future__ import annotations

import argparse

from stock_selection_fundamental.research.experiments import placeholder_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment placeholder.")
    parser.add_argument("--name", default="default_experiment")
    args = parser.parse_args()
    result = placeholder_experiments()
    print(f"Experiment '{args.name}' finished. rows={len(result)}")


if __name__ == "__main__":
    main()
