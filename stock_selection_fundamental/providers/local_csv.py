from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hk_stock_quant.data.local_csv import LocalCSVDataProvider as LegacyLocalCSV

from .base import DataProvider
from .mapping import (
    standardize_financials,
    standardize_price_history,
    standardize_release_calendar,
    standardize_security_master,
)


class LocalCSVDataProvider(DataProvider):
    """Compatibility provider using the existing curated CSV schema."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self._delegate = LegacyLocalCSV(self.base_dir)

    def get_security_master(self) -> pd.DataFrame:
        frame = self._delegate.get_security_master()
        return standardize_security_master(frame)

    def get_price_history(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._delegate.get_price_history(symbols=symbols, start=start, end=end)
        return standardize_price_history(frame)

    def get_financials(self, symbols: Sequence[str], as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        frame = self._delegate.get_financials(symbols=symbols, as_of_date=as_of_date)
        return standardize_financials(frame)

    def get_release_calendar(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._delegate.get_financial_release_calendar(symbols=symbols, start=start, end=end)
        return standardize_release_calendar(frame)

    def get_benchmark_history(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return self.get_price_history(symbols=[symbol], start=start, end=end)

    def get_trading_status(self, symbols: Sequence[str], date: str | pd.Timestamp) -> pd.DataFrame:
        frame = self._delegate.get_trading_status(symbols=symbols, date=date)
        output = standardize_price_history(frame)
        if "date" not in output.columns:
            output["date"] = pd.Timestamp(date)
        return output
