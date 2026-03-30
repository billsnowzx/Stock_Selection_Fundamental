from __future__ import annotations

import unittest

import pandas as pd

from stock_selection_fundamental.risk.attribution import brinson_lite_attribution, summarize_attribution


class AttributionTests(unittest.TestCase):
    def test_brinson_lite_reconciles_on_simple_case(self) -> None:
        nav_history = pd.DataFrame(
            [
                {"date": "2024-01-01", "nav": 1.0, "benchmark_nav": 1.0},
                {"date": "2024-01-02", "nav": 1.1, "benchmark_nav": 1.05},
            ]
        )
        holdings_history = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "portfolio_weight": 1.0},
                {"date": "2024-01-02", "symbol": "AAA", "portfolio_weight": 1.0},
            ]
        )
        price_history = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "close": 100.0},
                {"date": "2024-01-02", "symbol": "AAA", "close": 110.0},
                {"date": "2024-01-01", "symbol": "BBB", "close": 100.0},
                {"date": "2024-01-02", "symbol": "BBB", "close": 100.0},
            ]
        )
        security_master = pd.DataFrame(
            [
                {"symbol": "AAA", "industry_std": "TECH"},
                {"symbol": "BBB", "industry_std": "UTIL"},
            ]
        )

        attr = brinson_lite_attribution(
            nav_history=nav_history,
            holdings_history=holdings_history,
            price_history=price_history,
            security_master=security_master,
        )
        day2 = attr[attr["date"] == pd.Timestamp("2024-01-02")].iloc[0]
        self.assertAlmostEqual(float(day2["active_return"]), 0.05, places=9)
        self.assertAlmostEqual(float(day2["industry_component"]), 0.05, places=9)
        self.assertAlmostEqual(float(day2["selection_component"]), 0.0, places=9)
        self.assertAlmostEqual(float(day2["interaction_component"]), 0.0, places=9)
        self.assertAlmostEqual(float(day2["active_model_error"]), 0.0, places=9)
        self.assertAlmostEqual(float(day2["total_return_recon_error"]), 0.0, places=9)

        summary = summarize_attribution(attr)
        self.assertFalse(summary.empty)
        self.assertLess(float(summary["total_return_recon_error_abs_max"].iloc[0]), 1e-10)


if __name__ == "__main__":
    unittest.main()
