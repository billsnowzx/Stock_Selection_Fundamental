from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hk_stock_quant.demo_data import write_demo_dataset
from stock_selection_fundamental.config import load_config_bundle
from stock_selection_fundamental.providers.curation import prepare_curated_dataset
from stock_selection_fundamental.providers.local_csv import LocalCSVDataProvider


class CurationTests(unittest.TestCase):
    def test_prepare_curated_generates_visibility_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "sample"
            curated_dir = Path(temp_dir) / "curated"
            write_demo_dataset(data_dir, start="2021-01-01", end="2025-12-31", n_symbols=12, seed=17)

            bundle = load_config_bundle("configs/backtests/hk_top20.yaml")
            bundle.backtest["data_dir"] = str(data_dir)
            provider = LocalCSVDataProvider(data_dir)
            result = prepare_curated_dataset(provider=provider, config_bundle=bundle, output_dir=curated_dir, use_cache=False)

            self.assertTrue((curated_dir / "financials_visibility_snapshot.csv").exists())
            self.assertTrue((curated_dir / "curated_manifest.json").exists())
            self.assertGreater(result.manifest["n_price_rows"], 0)


if __name__ == "__main__":
    unittest.main()
