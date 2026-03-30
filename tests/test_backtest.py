from __future__ import annotations

import tempfile
import unittest

from hk_stock_quant.backtest import BacktestEngine
from hk_stock_quant.config import StrategyConfig
from hk_stock_quant.data.local_csv import LocalCSVDataProvider
from hk_stock_quant.demo_data import write_demo_dataset
from hk_stock_quant.strategy import FundamentalTopNStrategy


class BacktestEngineTests(unittest.TestCase):
    def test_backtest_runs_end_to_end_on_demo_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_demo_dataset(temp_dir, start="2021-01-01", end="2025-12-31", n_symbols=30, seed=11)
            provider = LocalCSVDataProvider(temp_dir)
            config = StrategyConfig(top_n=10, initial_capital=500000.0)
            strategy = FundamentalTopNStrategy(config)
            engine = BacktestEngine(config, strategy)
            result = engine.run(provider, start="2023-01-03", end="2025-12-31")

            self.assertFalse(result.nav_history.empty)
            self.assertFalse(result.selection_history.empty)
            self.assertFalse(result.trades.empty)
            self.assertIn("annualized_return", result.metrics)
            self.assertIn("ic_summary", result.factor_diagnostics)
            self.assertGreaterEqual(result.nav_history["nav"].iloc[-1], 0.8)


if __name__ == "__main__":
    unittest.main()
