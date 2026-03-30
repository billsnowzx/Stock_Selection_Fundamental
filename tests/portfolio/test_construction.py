from __future__ import annotations

import unittest

import pandas as pd

from stock_selection_fundamental.portfolio.construction import build_target_weights


class PortfolioConstructionTests(unittest.TestCase):
    def test_equal_weight_normalizes_and_respects_limits(self) -> None:
        selected = pd.DataFrame(
            [
                {"symbol": "A", "composite_score": 2.0},
                {"symbol": "B", "composite_score": 1.0},
                {"symbol": "C", "composite_score": 0.5},
            ]
        )
        weights = build_target_weights(
            selected=selected,
            method="equal_weight",
            max_weight=0.4,
            min_holdings=2,
            max_holdings=3,
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertLessEqual(float(weights.max()), 0.4)


if __name__ == "__main__":
    unittest.main()
