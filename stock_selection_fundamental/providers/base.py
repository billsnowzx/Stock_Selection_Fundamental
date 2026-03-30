from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class DataProvider(ABC):
    """Canonical data interface for research/backtest modules.

    Newly added optional methods return empty frames by default so legacy providers stay compatible.
    """

    @abstractmethod
    def get_security_master(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_price_history(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_financials(self, symbols: Sequence[str], as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_release_calendar(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_benchmark_history(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_trading_status(self, symbols: Sequence[str], date: str | pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError

    # Optional interfaces (phase-1+). Providers can override with real implementations.
    def get_adjustment_factors(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "symbol", "adj_factor"])

    def get_corporate_actions(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["ex_date", "symbol", "action_type", "ratio", "cash_dividend"])

    def get_lot_sizes(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "lot_size"])

    def get_industry_mapping(
        self,
        symbols: Sequence[str] | None = None,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "industry_std", "industry_source", "as_of_date"])

    def get_financial_history(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()
