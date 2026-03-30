from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hk_stock_quant.demo_data import write_demo_dataset
from stock_selection_fundamental.providers.local_csv import LocalCSVDataProvider


class VisibilityRuleTests(unittest.TestCase):
    def test_financial_visibility_uses_release_date(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            write_demo_dataset(data_dir, start="2021-01-01", end="2025-12-31", n_symbols=10, seed=23)

            release_path = data_dir / "release_calendar.csv"
            releases = pd.read_csv(release_path, parse_dates=["period_end", "release_date"])
            symbol = releases["symbol"].iloc[0]
            latest_idx = releases[releases["symbol"] == symbol]["period_end"].idxmax()
            latest_period = releases.loc[latest_idx, "period_end"]
            releases.loc[latest_idx, "release_date"] = pd.Timestamp("2026-12-31")
            releases.to_csv(release_path, index=False)

            provider = LocalCSVDataProvider(data_dir)
            snap = provider.get_financials(symbols=[symbol], as_of_date=pd.Timestamp("2025-12-31"))
            self.assertEqual(len(snap), 1)
            self.assertLess(snap.iloc[0]["period_end"], latest_period)


if __name__ == "__main__":
    unittest.main()
