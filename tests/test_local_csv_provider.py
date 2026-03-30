from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from hk_stock_quant.data.local_csv import LocalCSVDataProvider
from hk_stock_quant.demo_data import write_demo_dataset


class LocalCSVProviderTests(unittest.TestCase):
    def test_financial_snapshot_contains_latest_available_and_previous_year(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_demo_dataset(temp_dir, start="2021-01-01", end="2024-12-31", n_symbols=8, seed=7)
            provider = LocalCSVDataProvider(temp_dir)
            snapshot = provider.get_financials(["0001.HK"], pd.Timestamp("2024-08-31"))
            self.assertEqual(len(snapshot), 1)
            row = snapshot.iloc[0]
            self.assertLessEqual(row["release_date"], pd.Timestamp("2024-08-31"))
            self.assertFalse(pd.isna(row["prev_revenue"]))
            self.assertFalse(pd.isna(row["prev_net_income"]))


if __name__ == "__main__":
    unittest.main()
