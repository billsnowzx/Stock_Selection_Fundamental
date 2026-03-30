from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class DataProvider(ABC):
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
    def get_financials(
        self,
        symbols: Sequence[str],
        as_of_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_financial_release_calendar(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_trading_status(
        self,
        symbols: Sequence[str],
        date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        raise NotImplementedError

