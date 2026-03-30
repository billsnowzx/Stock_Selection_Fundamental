from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from hk_stock_quant.demo_data import write_demo_dataset
from stock_selection_fundamental.providers.local_csv import LocalCSVDataProvider


class ProviderFieldTests(unittest.TestCase):
    def test_local_csv_provider_returns_standard_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_demo_dataset(temp_dir, start="2021-01-01", end="2024-12-31", n_symbols=10, seed=3)
            provider = LocalCSVDataProvider(temp_dir)

            master = provider.get_security_master()
            prices = provider.get_price_history(start="2023-01-01", end="2023-12-31")
            releases = provider.get_release_calendar(start="2023-01-01", end="2024-12-31")
            snapshot = provider.get_financials(symbols=["0001.HK"], as_of_date=pd.Timestamp("2024-08-31"))
            financial_history = provider.get_financial_history(symbols=["0001.HK"])
            lot_sizes = provider.get_lot_sizes(["0001.HK"])
            industry = provider.get_industry_mapping(["0001.HK"], as_of_date=pd.Timestamp("2024-08-31"))
            adjustments = provider.get_adjustment_factors(["0001.HK"])
            actions = provider.get_corporate_actions(["0001.HK"])

            self.assertTrue({"symbol", "list_date", "market", "exchange"}.issubset(master.columns))
            self.assertTrue({"date", "symbol", "open", "close", "is_suspended"}.issubset(prices.columns))
            self.assertTrue({"symbol", "period_end", "release_date"}.issubset(releases.columns))
            self.assertTrue({"symbol", "period_end", "revenue", "net_income"}.issubset(snapshot.columns))
            self.assertTrue({"symbol", "period_end"}.issubset(financial_history.columns))
            self.assertTrue({"symbol", "lot_size"}.issubset(lot_sizes.columns))
            self.assertTrue({"symbol", "industry_std"}.issubset(industry.columns))
            self.assertTrue({"date", "symbol", "adj_factor"}.issubset(adjustments.columns))
            self.assertTrue({"ex_date", "symbol", "action_type", "ratio", "cash_dividend"}.issubset(actions.columns))
            self.assertLessEqual(snapshot.iloc[0]["release_date"], pd.Timestamp("2024-08-31"))


if __name__ == "__main__":
    unittest.main()
