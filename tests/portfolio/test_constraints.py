from __future__ import annotations

import unittest

import pandas as pd

from stock_selection_fundamental.portfolio.constraints import enforce_turnover_cap
from stock_selection_fundamental.portfolio.turnover import portfolio_turnover


class PortfolioConstraintTests(unittest.TestCase):
    def test_turnover_cap_blends_target(self) -> None:
        previous = pd.Series({"AAA": 1.0})
        target = pd.Series({"BBB": 1.0})
        capped = enforce_turnover_cap(previous=previous, target=target, max_turnover=0.2)
        self.assertAlmostEqual(float(capped.sum()), 1.0, places=6)
        self.assertLessEqual(portfolio_turnover(previous, capped), 0.200001)
        self.assertGreater(float(capped.get("AAA", 0.0)), 0.0)
        self.assertGreater(float(capped.get("BBB", 0.0)), 0.0)

    def test_turnover_cap_none_keeps_target(self) -> None:
        previous = pd.Series({"AAA": 1.0})
        target = pd.Series({"AAA": 0.2, "BBB": 0.8})
        uncapped = enforce_turnover_cap(previous=previous, target=target, max_turnover=None)
        self.assertAlmostEqual(float(uncapped.get("AAA", 0.0)), 0.2, places=6)
        self.assertAlmostEqual(float(uncapped.get("BBB", 0.0)), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
