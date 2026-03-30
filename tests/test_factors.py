from __future__ import annotations

import unittest

import pandas as pd

from hk_stock_quant.config import StrategyConfig
from hk_stock_quant.factors import FactorScorer, default_factor_definitions


class FactorScorerTests(unittest.TestCase):
    def test_scores_positive_and_negative_factors_correctly(self) -> None:
        financials = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "revenue": 100.0,
                    "prev_revenue": 90.0,
                    "net_income": 18.0,
                    "prev_net_income": 15.0,
                    "nopat": 20.0,
                    "invested_capital": 100.0,
                    "total_liabilities": 50.0,
                    "operating_cashflow": 25.0,
                    "free_cashflow": 17.0,
                },
                {
                    "symbol": "B",
                    "revenue": 100.0,
                    "prev_revenue": 98.0,
                    "net_income": 9.0,
                    "prev_net_income": 8.0,
                    "nopat": 10.0,
                    "invested_capital": 100.0,
                    "total_liabilities": 90.0,
                    "operating_cashflow": 15.0,
                    "free_cashflow": 6.0,
                },
            ]
        )
        market = pd.DataFrame(
            [
                {"symbol": "A", "date": pd.Timestamp("2024-01-31"), "close": 10.0},
                {"symbol": "B", "date": pd.Timestamp("2024-01-31"), "close": 8.0},
            ]
        )
        scorer = FactorScorer(StrategyConfig(min_factors_required=1), default_factor_definitions())
        scored = scorer.score(financials, market)
        scores = scored.set_index("symbol")
        self.assertGreater(scores.at["A", "composite_score"], scores.at["B", "composite_score"])
        self.assertGreater(scores.at["A", "debt_to_cashflow_score"], scores.at["B", "debt_to_cashflow_score"])
        self.assertEqual(int(scores.at["A", "threshold_pass_count"]), 6)


if __name__ == "__main__":
    unittest.main()

