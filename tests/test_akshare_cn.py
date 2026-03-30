from __future__ import annotations

import unittest

import pandas as pd

from hk_stock_quant.data.akshare_cn import AkshareCNDataProvider


class AkshareCNAdapterTests(unittest.TestCase):
    def test_normalize_symbol(self) -> None:
        self.assertEqual(AkshareCNDataProvider._normalize_symbol("600519"), "600519.SH")
        self.assertEqual(AkshareCNDataProvider._normalize_symbol("000001.SZ"), "000001.SZ")
        self.assertEqual(AkshareCNDataProvider._normalize_symbol("sz300750"), "300750.SZ")
        self.assertIsNone(AkshareCNDataProvider._normalize_symbol("920188"))

    def test_normalize_index_symbol(self) -> None:
        self.assertEqual(AkshareCNDataProvider._normalize_index_symbol("000300"), "sh000300")
        self.assertEqual(AkshareCNDataProvider._normalize_index_symbol("sz399001"), "sz399001")

    def test_extract_statement_values_and_sum(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "报告日": "2024-12-31",
                    "负债合计": 300.0,
                    "流动负债合计": 120.0,
                    "资产总计": 900.0,
                    "经营活动产生的现金流量净额": 100.0,
                    "购建固定资产、无形资产和其他长期资产所支付的现金": 25.0,
                }
            ]
        )
        values = AkshareCNDataProvider._extract_statement_values(
            frame,
            {
                "total_liabilities": ["负债合计"],
                "current_liabilities": ["流动负债合计"],
                "total_assets": ["资产总计"],
                "operating_cashflow": ["经营活动产生的现金流量净额"],
            },
        )
        capex = AkshareCNDataProvider._extract_statement_sum(
            frame,
            "capital_expenditure",
            ["购建固定资产、无形资产和其他长期资产所支付的现金"],
        )
        self.assertEqual(float(values.iloc[0]["total_liabilities"]), 300.0)
        self.assertEqual(float(values.iloc[0]["current_liabilities"]), 120.0)
        self.assertEqual(float(values.iloc[0]["total_assets"]), 900.0)
        self.assertEqual(float(values.iloc[0]["operating_cashflow"]), 100.0)
        self.assertEqual(float(capex.iloc[0]["capital_expenditure"]), 25.0)

    def test_estimate_release_date(self) -> None:
        annual = AkshareCNDataProvider._estimate_release_date(pd.Timestamp("2024-12-31"), "年报")
        q3 = AkshareCNDataProvider._estimate_release_date(pd.Timestamp("2024-09-30"), "三季报")
        self.assertEqual(annual, pd.Timestamp("2025-03-31"))
        self.assertEqual(q3, pd.Timestamp("2024-11-14"))


if __name__ == "__main__":
    unittest.main()
