from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hk_stock_quant.demo_data import write_demo_dataset
from stock_selection_fundamental.backtest.engine import BacktestEngine
from stock_selection_fundamental.config import ConfigBundle
from stock_selection_fundamental.providers.local_csv import LocalCSVDataProvider
from stock_selection_fundamental.reporting.export_csv import export_csv_outputs


class IntegrationPipelineTests(unittest.TestCase):
    def test_demo_data_pipeline_runs_and_exports_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            output_dir = Path(temp_dir) / "outputs"
            write_demo_dataset(data_dir, start="2021-01-01", end="2025-12-31", n_symbols=25, seed=9)

            bundle = ConfigBundle(
                market={
                    "benchmark_symbol": "^HSI",
                    "universe": {
                        "board": "MAIN",
                        "security_type": "EQUITY",
                        "excluded_security_types": [],
                        "min_listing_days": 252,
                        "liquidity_lookback_days": 20,
                        "min_avg_turnover": 0,
                        "exclude_st": False,
                    },
                },
                strategy={
                    "factor_weights": {
                        "roic": 1.0,
                        "net_margin": 1.0,
                        "debt_to_cashflow": 1.0,
                        "revenue_growth_yoy": 1.0,
                        "net_income_growth_yoy": 1.0,
                        "fcf_conversion": 1.0,
                    },
                    "transform": {"method": "zscore", "winsorize_limits": [0.05, 0.95], "by_industry": False},
                    "min_factors_required": 4,
                    "selection": {"top_n": 10, "top_percentile": None, "min_selection": 5},
                    "portfolio": {"weight_method": "equal_weight", "max_single_weight": 0.2, "min_holdings": 5, "max_holdings": 15},
                    "quantiles": 5,
                },
                backtest={
                    "start": "2023-01-03",
                    "end": "2025-12-31",
                    "initial_capital": 1_000_000,
                    "benchmark_symbol": "^HSI",
                    "rebalance_frequency": "M",
                    "costs": {"transaction_cost_bps": 20, "slippage_bps": 10, "minimum_fee": 0},
                },
                risk={},
            )

            provider = LocalCSVDataProvider(data_dir)
            engine = BacktestEngine(bundle)
            artifacts = engine.run(provider)

            self.assertFalse(artifacts.nav_history.empty)
            self.assertFalse(artifacts.selection_history.empty)
            self.assertIn("annualized_return", artifacts.metrics)

            export_csv_outputs(artifacts, output_dir, bundle.as_dict())
            self.assertTrue((output_dir / "nav_history.csv").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "ic_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
