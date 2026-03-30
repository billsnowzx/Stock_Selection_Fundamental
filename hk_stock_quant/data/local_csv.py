from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .provider import DataProvider


class LocalCSVDataProvider(DataProvider):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self._security_master: pd.DataFrame | None = None
        self._price_history: pd.DataFrame | None = None
        self._financials: pd.DataFrame | None = None
        self._release_calendar: pd.DataFrame | None = None

    def get_security_master(self) -> pd.DataFrame:
        if self._security_master is None:
            path = self.base_dir / "security_master.csv"
            frame = pd.read_csv(path, parse_dates=["list_date", "delist_date"])
            self._security_master = frame.sort_values("symbol").reset_index(drop=True)
        return self._security_master.copy()

    def get_price_history(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if self._price_history is None:
            path = self.base_dir / "price_history.csv"
            frame = pd.read_csv(path, parse_dates=["date"])
            frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
            self._price_history = frame

        frame = self._price_history
        if symbols is not None:
            frame = frame[frame["symbol"].isin(symbols)]
        if start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        return frame.copy()

    def get_financial_release_calendar(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._load_release_calendar()
        if symbols is not None:
            frame = frame[frame["symbol"].isin(symbols)]
        if start is not None:
            frame = frame[frame["release_date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["release_date"] <= pd.Timestamp(end)]
        return frame.copy()

    def get_trading_status(
        self,
        symbols: Sequence[str],
        date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        query_date = pd.Timestamp(date)
        prices = self.get_price_history(symbols=symbols, start=query_date, end=query_date)
        if prices.empty:
            return pd.DataFrame(
                {
                    "symbol": list(symbols),
                    "date": query_date,
                    "is_tradable": False,
                    "is_suspended": True,
                    "open": float("nan"),
                    "close": float("nan"),
                    "turnover": float("nan"),
                }
            )

        prices = prices[["symbol", "date", "is_suspended", "open", "close", "turnover"]].copy()
        prices["is_tradable"] = (~prices["is_suspended"].fillna(True)) & prices["open"].gt(0)
        missing = sorted(set(symbols) - set(prices["symbol"]))
        if missing:
            missing_frame = pd.DataFrame(
                {
                    "symbol": missing,
                    "date": query_date,
                    "is_suspended": True,
                    "open": float("nan"),
                    "close": float("nan"),
                    "turnover": float("nan"),
                    "is_tradable": False,
                }
            )
            prices = pd.concat([prices, missing_frame], ignore_index=True)
        return prices.sort_values("symbol").reset_index(drop=True)

    def get_financials(
        self,
        symbols: Sequence[str],
        as_of_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of_date)
        frame = self._load_financials_with_release()
        frame = frame[frame["symbol"].isin(symbols)]
        frame = frame[frame["release_date"] <= as_of].copy()
        if frame.empty:
            return pd.DataFrame(columns=["symbol"])

        latest = (
            frame.sort_values(["symbol", "release_date", "period_end"])
            .groupby("symbol", group_keys=False)
            .tail(1)
            .reset_index(drop=True)
        )

        enriched_rows: list[pd.Series] = []
        for _, row in latest.iterrows():
            history = frame[frame["symbol"] == row["symbol"]].sort_values("period_end")
            target_period = row["period_end"] - pd.DateOffset(years=1)
            prior = history[history["period_end"] <= target_period].tail(1)
            row = row.copy()
            if not prior.empty:
                prev_row = prior.iloc[0]
                row["prev_period_end"] = prev_row["period_end"]
                row["prev_revenue"] = prev_row.get("revenue")
                row["prev_net_income"] = prev_row.get("net_income")
            else:
                row["prev_period_end"] = pd.NaT
                row["prev_revenue"] = float("nan")
                row["prev_net_income"] = float("nan")
            row["age_days"] = (as_of - row["release_date"]).days
            enriched_rows.append(row)

        snapshot = pd.DataFrame(enriched_rows)
        if "free_cashflow" not in snapshot.columns and {
            "operating_cashflow",
            "capital_expenditure",
        }.issubset(snapshot.columns):
            snapshot["free_cashflow"] = (
                snapshot["operating_cashflow"] - snapshot["capital_expenditure"]
            )
        if "nopat" not in snapshot.columns:
            if {"ebit", "effective_tax_rate"}.issubset(snapshot.columns):
                snapshot["nopat"] = snapshot["ebit"] * (1 - snapshot["effective_tax_rate"])
            else:
                snapshot["nopat"] = snapshot.get("net_income")
        return snapshot.sort_values("symbol").reset_index(drop=True)

    def _load_financials_with_release(self) -> pd.DataFrame:
        if self._financials is None:
            path = self.base_dir / "financials.csv"
            frame = pd.read_csv(path, parse_dates=["period_end"])
            self._financials = frame

        frame = self._financials.copy()
        if "release_date" in frame.columns:
            frame["release_date"] = pd.to_datetime(frame["release_date"])
            return frame

        releases = self._load_release_calendar()
        return frame.merge(releases, on=["symbol", "period_end"], how="left")

    def _load_release_calendar(self) -> pd.DataFrame:
        if self._release_calendar is None:
            path = self.base_dir / "release_calendar.csv"
            frame = pd.read_csv(path, parse_dates=["period_end", "release_date"])
            self._release_calendar = frame.sort_values(
                ["symbol", "release_date", "period_end"]
            ).reset_index(drop=True)
        return self._release_calendar.copy()

