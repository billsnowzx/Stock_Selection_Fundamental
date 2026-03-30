from __future__ import annotations

import unittest

from stock_selection_fundamental.backtest.cost_model import CostModel


class CostModelTests(unittest.TestCase):
    def test_cost_and_slippage(self) -> None:
        model = CostModel(transaction_cost_bps=20.0, slippage_bps=10.0, minimum_fee=5.0)
        self.assertAlmostEqual(model.apply_buy_slippage(100.0), 100.1)
        self.assertAlmostEqual(model.apply_sell_slippage(100.0), 99.9)
        self.assertAlmostEqual(model.fee(1000.0), 5.0)
        self.assertAlmostEqual(model.fee(100000.0), 200.0)


if __name__ == "__main__":
    unittest.main()
