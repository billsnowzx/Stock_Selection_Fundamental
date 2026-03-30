from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stock_selection_fundamental.research.regression import compare_baseline_metrics, freeze_baseline_metrics


class RegressionBaselineTests(unittest.TestCase):
    def test_freeze_and_compare_baseline(self) -> None:
        metrics = {
            "total_return": 0.12,
            "annualized_return": 0.08,
            "annualized_volatility": 0.15,
            "sharpe": 0.6,
            "max_drawdown": -0.1,
            "turnover": 0.3,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline = Path(temp_dir) / "baseline_metrics.json"
            freeze_baseline_metrics(metrics, baseline)
            result_ok = compare_baseline_metrics(baseline, {**metrics, "total_return": 0.121}, tolerance_bps=50)
            result_fail = compare_baseline_metrics(baseline, {**metrics, "total_return": 0.20}, tolerance_bps=50)
            self.assertTrue(result_ok.passed)
            self.assertFalse(result_fail.passed)


if __name__ == "__main__":
    unittest.main()
