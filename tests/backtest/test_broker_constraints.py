from __future__ import annotations

import unittest

import pandas as pd

from stock_selection_fundamental.backtest.broker import build_price_tables, execute_rebalance
from stock_selection_fundamental.backtest.cost_model import CostModel


class BrokerConstraintTests(unittest.TestCase):
    def test_rebalance_respects_lot_size_and_min_notional(self) -> None:
        prices = pd.DataFrame(
            [
                {"date": "2024-01-31", "symbol": "AAA", "open": 10.0, "close": 10.0, "is_suspended": False},
                {"date": "2024-01-31", "symbol": "BBB", "open": 20.0, "close": 20.0, "is_suspended": False},
            ]
        )
        prices["date"] = pd.to_datetime(prices["date"])
        prices["high"] = prices["open"]
        prices["low"] = prices["open"]
        prices["volume"] = 1_000_000
        prices["turnover"] = prices["open"] * prices["volume"]
        tables = build_price_tables(prices)

        holdings = {}
        target = pd.Series({"AAA": 0.5, "BBB": 0.5})
        lot_sizes = pd.Series({"AAA": 100, "BBB": 100})
        holdings, cash, trades = execute_rebalance(
            date=pd.Timestamp("2024-01-31"),
            target_weights=target,
            holdings=holdings,
            cash=10000.0,
            price_tables=tables,
            cost_model=CostModel(transaction_cost_bps=0, slippage_bps=0),
            lot_sizes=lot_sizes,
            min_trade_notional=2000.0,
        )
        self.assertFalse(trades.empty)
        self.assertTrue((trades["shares"] % 100 == 0).all())
        self.assertTrue((trades["notional"] >= 2000.0).all())
        self.assertGreaterEqual(cash, 0.0)


if __name__ == "__main__":
    unittest.main()
