from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hk_stock_quant.demo_data import write_demo_dataset
from stock_selection_fundamental.backtest.engine import BacktestEngine
from stock_selection_fundamental.config import load_config_bundle
from stock_selection_fundamental.providers.local_csv import LocalCSVDataProvider


class CorporateActionsIntegrationTests(unittest.TestCase):
    def test_corporate_actions_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "sample"
            write_demo_dataset(data_dir, start="2021-01-01", end="2025-12-31", n_symbols=25, seed=41)

            master = pd.read_csv(data_dir / "security_master.csv")
            symbols = master["symbol"].head(10).tolist()
            actions = pd.DataFrame(
                {
                    "ex_date": ["2024-06-28"] * len(symbols),
                    "symbol": symbols,
                    "action_type": ["cash_dividend"] * len(symbols),
                    "ratio": [1.0] * len(symbols),
                    "cash_dividend": [0.05] * len(symbols),
                }
            )
            actions.to_csv(data_dir / "corporate_actions.csv", index=False)

            bundle = load_config_bundle("configs/backtests/hk_top20.yaml")
            bundle.backtest["data_dir"] = str(data_dir)
            artifacts = BacktestEngine(bundle).run(LocalCSVDataProvider(data_dir))
            ledger = artifacts.research_outputs.get("corporate_action_ledger")
            self.assertIsNotNone(ledger)
            self.assertFalse(ledger.empty)
            self.assertIn("cash_delta", ledger.columns)


if __name__ == "__main__":
    unittest.main()
