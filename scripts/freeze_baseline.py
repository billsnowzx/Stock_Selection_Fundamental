from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_selection_fundamental.research.regression import freeze_baseline_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze baseline metrics JSON.")
    parser.add_argument("--metrics-file", default="outputs/hk_top20/latest/metrics.json")
    parser.add_argument("--output", default="tests/baseline/baseline_metrics.json")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
    path = freeze_baseline_metrics(metrics=metrics, output_path=args.output)
    print(f"Baseline frozen: {path.resolve()}")


if __name__ == "__main__":
    main()
