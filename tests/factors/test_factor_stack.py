from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_selection_fundamental.factors.registry import build_factor_panel
from stock_selection_fundamental.factors.transforms import winsorize, zscore
from stock_selection_fundamental.signals.composite_score import compute_composite_score


class FactorStackTests(unittest.TestCase):
    def test_factor_formula_and_direction(self) -> None:
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
                    "industry": "Tech",
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
                    "industry": "Tech",
                },
            ]
        )
        panel = build_factor_panel(financials, transform="zscore", winsor_limits=(0.0, 1.0))
        scored = compute_composite_score(
            panel,
            weights={
                "roic": 1.0,
                "net_margin": 1.0,
                "debt_to_cashflow": 1.0,
                "revenue_growth_yoy": 1.0,
                "net_income_growth_yoy": 1.0,
                "fcf_conversion": 1.0,
            },
            min_factors_required=1,
        )
        scores = scored.set_index("symbol")
        self.assertGreater(scores.at["A", "composite_score"], scores.at["B", "composite_score"])
        self.assertGreater(scores.at["A", "debt_to_cashflow_score"], scores.at["B", "debt_to_cashflow_score"])

    def test_winsorize_and_zscore_behaviour(self) -> None:
        series = pd.Series([1.0, 2.0, 1000.0, np.nan])
        clipped = winsorize(series, 0.0, 0.75)
        self.assertLess(float(clipped.dropna().max()), 1000.0)
        z = zscore(pd.Series([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(z.mean()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
