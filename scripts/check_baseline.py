from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_selection_fundamental.research.regression import compare_baseline_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Check baseline metrics drift.")
    parser.add_argument("--baseline", default="tests/baseline/baseline_metrics.json")
    parser.add_argument("--metrics-file", default="outputs/hk_top20/latest/metrics.json")
    parser.add_argument("--tolerance-bps", type=float, default=200.0)
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
    result = compare_baseline_metrics(args.baseline, metrics, tolerance_bps=args.tolerance_bps)
    print(result.details.to_string(index=False))
    if not result.passed:
        raise SystemExit("Baseline regression check failed.")
    print("Baseline regression check passed.")


if __name__ == "__main__":
    main()
