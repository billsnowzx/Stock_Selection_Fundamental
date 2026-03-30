from __future__ import annotations

import pandas as pd

from .config import StrategyConfig
from .data.provider import DataProvider


class UniverseBuilder:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def build(self, provider: DataProvider, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of_date)
        master = provider.get_security_master()
        master = master[
            (master["board"] == self.config.board)
            & (master["security_type"] == self.config.security_type)
            & (~master["security_type"].isin(self.config.excluded_security_types))
            & (master["list_date"] <= as_of)
            & (master["delist_date"].isna() | (master["delist_date"] >= as_of))
        ].copy()
        if master.empty:
            return master

        price_history = provider.get_price_history(symbols=master["symbol"], end=as_of)
        trading_days = (
            price_history.groupby("symbol")["date"].nunique().rename("trading_days_listed")
        )
        master = master.merge(trading_days, on="symbol", how="left")
        master["trading_days_listed"] = master["trading_days_listed"].fillna(0)
        master = master[master["trading_days_listed"] >= self.config.min_listing_days]
        if master.empty:
            return master

        trading_status = provider.get_trading_status(master["symbol"].tolist(), as_of)
        master = master.merge(
            trading_status[["symbol", "is_tradable", "is_suspended", "close", "turnover"]],
            on="symbol",
            how="left",
        )
        master["is_tradable"] = master["is_tradable"].fillna(False)
        master["is_suspended"] = master["is_suspended"].fillna(True)
        return master[master["is_tradable"]].sort_values("symbol").reset_index(drop=True)

