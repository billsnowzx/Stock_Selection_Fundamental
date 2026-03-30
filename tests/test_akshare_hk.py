from __future__ import annotations

import unittest

import pandas as pd

from hk_stock_quant.data.akshare_hk import AkshareHKDataProvider


class AkshareHKAdapterTests(unittest.TestCase):
    def test_estimate_release_date_uses_conservative_lag(self) -> None:
        annual = AkshareHKDataProvider._estimate_release_date(pd.Timestamp("2024-12-31"), "001")
        q3 = AkshareHKDataProvider._estimate_release_date(pd.Timestamp("2024-09-30"), "004")
        self.assertEqual(annual, pd.Timestamp("2025-03-31"))
        self.assertEqual(q3, pd.Timestamp("2024-11-14"))

    def test_extract_statement_values_and_sum(self) -> None:
        frame = pd.DataFrame(
            [
                {"REPORT_DATE": "2024-12-31", "STD_ITEM_NAME": "总负债", "AMOUNT": 300.0},
                {"REPORT_DATE": "2024-12-31", "STD_ITEM_NAME": "经营业务现金净额", "AMOUNT": 100.0},
                {"REPORT_DATE": "2024-12-31", "STD_ITEM_NAME": "购建固定资产", "AMOUNT": 20.0},
                {"REPORT_DATE": "2024-12-31", "STD_ITEM_NAME": "购建无形资产及其他资产", "AMOUNT": 5.0},
            ]
        )
        values = AkshareHKDataProvider._extract_statement_values(frame, {"total_liabilities": ["总负债"]})
        capex = AkshareHKDataProvider._extract_statement_sum(frame, "capital_expenditure", ["购建固定资产", "购建无形资产及其他资产"])
        self.assertEqual(float(values.iloc[0]["total_liabilities"]), 300.0)
        self.assertEqual(float(capex.iloc[0]["capital_expenditure"]), 25.0)

    def test_normalize_symbol(self) -> None:
        self.assertEqual(AkshareHKDataProvider._normalize_symbol("700"), "00700.HK")
        self.assertEqual(AkshareHKDataProvider._normalize_symbol("00700.HK"), "00700.HK")


if __name__ == "__main__":
    unittest.main()
