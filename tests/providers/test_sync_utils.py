from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_selection_fundamental.providers.sync_utils import merge_sync_outputs, resolve_incremental_window


class SyncUtilsTests(unittest.TestCase):
    def test_resolve_incremental_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            w = resolve_incremental_window(root, end="2025-12-31", default_start="2020-01-01")
            self.assertTrue(w.should_sync)
            self.assertEqual(w.sync_start, "2020-01-01")

            prices = pd.DataFrame({"date": ["2025-12-30"], "symbol": ["0001.HK"], "close": [10.0]})
            prices.to_csv(root / "price_history.csv", index=False)
            w2 = resolve_incremental_window(root, end="2025-12-31", default_start="2020-01-01")
            self.assertTrue(w2.should_sync)
            self.assertEqual(w2.sync_start, "2025-12-31")

    def test_merge_sync_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "base"
            new = Path(temp_dir) / "new"
            base.mkdir()
            new.mkdir()

            pd.DataFrame({"symbol": ["A"], "name": ["x"]}).to_csv(base / "security_master.csv", index=False)
            pd.DataFrame({"symbol": ["A"], "name": ["x2"]}).to_csv(new / "security_master.csv", index=False)

            pd.DataFrame({"date": ["2024-01-01"], "symbol": ["A"], "close": [10]}).to_csv(base / "price_history.csv", index=False)
            pd.DataFrame({"date": ["2024-01-02"], "symbol": ["A"], "close": [11]}).to_csv(new / "price_history.csv", index=False)

            pd.DataFrame({"symbol": ["A"], "period_end": ["2023-12-31"], "x": [1]}).to_csv(base / "financials.csv", index=False)
            pd.DataFrame({"symbol": ["A"], "period_end": ["2024-03-31"], "x": [2]}).to_csv(new / "financials.csv", index=False)

            pd.DataFrame({"symbol": ["A"], "period_end": ["2023-12-31"], "release_date": ["2024-03-01"]}).to_csv(base / "release_calendar.csv", index=False)
            pd.DataFrame({"symbol": ["A"], "period_end": ["2024-03-31"], "release_date": ["2024-05-01"]}).to_csv(new / "release_calendar.csv", index=False)

            merge_sync_outputs(base, new)
            merged_prices = pd.read_csv(base / "price_history.csv")
            self.assertEqual(len(merged_prices), 2)
            self.assertEqual(merged_prices["date"].max(), "2024-01-02")


if __name__ == "__main__":
    unittest.main()
