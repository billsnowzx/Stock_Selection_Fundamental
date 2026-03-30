from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from stock_selection_fundamental.providers.akshare_cn import AkshareCNDataProvider
from stock_selection_fundamental.providers.akshare_hk import AkshareHKDataProvider


class AkshareIncrementalTests(unittest.TestCase):
    def test_hk_skip_when_up_to_date(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            pd.DataFrame(
                [
                    {"date": "2025-12-31", "symbol": "0001.HK", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "turnover": 1, "is_suspended": False}
                ]
            ).to_csv(out / "price_history.csv", index=False)

            with patch("stock_selection_fundamental.providers.akshare_hk.LegacyHKProvider.sync_to_local_dataset") as mocked:
                AkshareHKDataProvider.sync_to_local_dataset(
                    output_dir=out,
                    start="2024-01-01",
                    end="2025-12-31",
                    symbols=["0001.HK"],
                    incremental=True,
                )
                mocked.assert_not_called()
            self.assertTrue((out / "sync_checkpoint.json").exists())

    def test_cn_incremental_calls_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            pd.DataFrame(
                [
                    {"date": "2025-12-30", "symbol": "000001.SZ", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "turnover": 1, "is_suspended": False}
                ]
            ).to_csv(out / "price_history.csv", index=False)
            pd.DataFrame([{"symbol": "000001.SZ", "name": "X"}]).to_csv(out / "security_master.csv", index=False)
            pd.DataFrame([{"symbol": "000001.SZ", "period_end": "2025-09-30"}]).to_csv(out / "financials.csv", index=False)
            pd.DataFrame([{"symbol": "000001.SZ", "period_end": "2025-09-30", "release_date": "2025-10-31"}]).to_csv(out / "release_calendar.csv", index=False)

            def fake_sync(**kwargs):
                tmp = Path(kwargs["output_dir"])
                pd.DataFrame([{"symbol": "000001.SZ", "name": "X"}]).to_csv(tmp / "security_master.csv", index=False)
                pd.DataFrame(
                    [
                        {"date": "2025-12-31", "symbol": "000001.SZ", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 1, "turnover": 1, "is_suspended": False}
                    ]
                ).to_csv(tmp / "price_history.csv", index=False)
                pd.DataFrame([{"symbol": "000001.SZ", "period_end": "2025-12-31"}]).to_csv(tmp / "financials.csv", index=False)
                pd.DataFrame([{"symbol": "000001.SZ", "period_end": "2025-12-31", "release_date": "2026-02-01"}]).to_csv(tmp / "release_calendar.csv", index=False)
                return tmp

            with patch("stock_selection_fundamental.providers.akshare_cn.LegacyCNProvider.sync_to_local_dataset", side_effect=fake_sync) as mocked:
                AkshareCNDataProvider.sync_to_local_dataset(
                    output_dir=out,
                    start="2024-01-01",
                    end="2025-12-31",
                    symbols=["000001.SZ"],
                    incremental=True,
                )
                mocked.assert_called_once()
            merged = pd.read_csv(out / "price_history.csv")
            self.assertEqual(merged["date"].max(), "2025-12-31")


if __name__ == "__main__":
    unittest.main()
