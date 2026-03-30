from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hk_stock_quant.demo_data import write_demo_dataset
from stock_selection_fundamental.backtest.engine import BacktestEngine
from stock_selection_fundamental.config import load_config_bundle
from stock_selection_fundamental.providers.local_csv import LocalCSVDataProvider


class AttributionIntegrationTests(unittest.TestCase):
    def test_attribution_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "sample"
            write_demo_dataset(data_dir, start="2021-01-01", end="2025-12-31", n_symbols=20, seed=31)
            bundle = load_config_bundle("configs/backtests/hk_top20.yaml")
            bundle.backtest["data_dir"] = str(data_dir)
            artifacts = BacktestEngine(bundle).run(LocalCSVDataProvider(data_dir))
            attr = artifacts.research_outputs.get("attribution_daily")
            self.assertIsNotNone(attr)
            self.assertFalse(attr.empty)
            self.assertTrue({"date", "active_return", "industry_component", "selection_component"}.issubset(attr.columns))


if __name__ == "__main__":
    unittest.main()
