from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hk_stock_quant.data.local_csv import LocalCSVDataProvider as LegacyLocalCSV

from .base import DataProvider
from .mapping import (
    normalize_industry_label,
    standardize_financials,
    standardize_price_history,
    standardize_release_calendar,
    standardize_security_master,
)


class LocalCSVDataProvider(DataProvider):
    """Compatibility provider using existing curated CSV schema plus optional extended files."""

    def __init__(self, base_dir: str | Path, mapping_key: str = "local_csv_v1"):
        self.base_dir = Path(base_dir)
        self.mapping_key = mapping_key
        self._delegate = LegacyLocalCSV(self.base_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def get_security_master(self) -> pd.DataFrame:
        frame = self._delegate.get_security_master()
        output = standardize_security_master(frame, mapping_key=self.mapping_key)
        if "industry_std" not in output.columns:
            output["industry_std"] = output.get("industry", "UNKNOWN")
            output["industry_std"] = output["industry_std"].map(normalize_industry_label)
        return output

    def get_price_history(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._delegate.get_price_history(symbols=symbols, start=start, end=end)
        return standardize_price_history(frame, mapping_key=self.mapping_key)

    def get_financials(self, symbols: Sequence[str], as_of_date: str | pd.Timestamp) -> pd.DataFrame:
        frame = self._delegate.get_financials(symbols=symbols, as_of_date=as_of_date)
        return standardize_financials(frame, mapping_key=self.mapping_key)

    def get_financial_history(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._load_optional_csv(
            filename="financials.csv",
            default=pd.DataFrame(columns=["symbol", "period_end"]),
            parse_dates=["period_end", "release_date"],
        )
        frame = standardize_financials(frame, mapping_key=self.mapping_key)
        if "release_date" not in frame.columns or frame["release_date"].isna().all():
            release = self.get_release_calendar(symbols=symbols, start=start, end=end)
            frame = frame.drop(columns=["release_date"], errors="ignore").merge(
                release[["symbol", "period_end", "release_date"]],
                on=["symbol", "period_end"],
                how="left",
            )
        output = frame.copy()
        if symbols is not None:
            output = output[output["symbol"].isin(symbols)]
        if start is not None:
            output = output[output["period_end"] >= pd.Timestamp(start)]
        if end is not None:
            output = output[output["period_end"] <= pd.Timestamp(end)]
        return output.sort_values(["symbol", "period_end"]).reset_index(drop=True)

    def get_release_calendar(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._delegate.get_financial_release_calendar(symbols=symbols, start=start, end=end)
        frame = standardize_release_calendar(frame, mapping_key=self.mapping_key)
        if symbols is not None:
            frame = frame[frame["symbol"].isin(symbols)]
        if start is not None:
            frame = frame[frame["release_date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["release_date"] <= pd.Timestamp(end)]
        return frame.reset_index(drop=True)

    def get_benchmark_history(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return self.get_price_history(symbols=[symbol], start=start, end=end)

    def get_trading_status(self, symbols: Sequence[str], date: str | pd.Timestamp) -> pd.DataFrame:
        frame = self._delegate.get_trading_status(symbols=symbols, date=date)
        output = standardize_price_history(frame, mapping_key=self.mapping_key)
        if "date" not in output.columns:
            output["date"] = pd.Timestamp(date)
        return output

    def get_adjustment_factors(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._load_optional_csv(
            filename="adjustment_factors.csv",
            default=pd.DataFrame(columns=["date", "symbol", "adj_factor"]),
            parse_dates=["date"],
        )
        if frame.empty:
            return frame
        output = frame.copy()
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        output["adj_factor"] = pd.to_numeric(output.get("adj_factor"), errors="coerce").fillna(1.0)
        if symbols is not None:
            output = output[output["symbol"].isin(symbols)]
        if start is not None:
            output = output[output["date"] >= pd.Timestamp(start)]
        if end is not None:
            output = output[output["date"] <= pd.Timestamp(end)]
        return output.sort_values(["date", "symbol"]).reset_index(drop=True)

    def get_corporate_actions(
        self,
        symbols: Sequence[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._load_optional_csv(
            filename="corporate_actions.csv",
            default=pd.DataFrame(columns=["ex_date", "symbol", "action_type", "ratio", "cash_dividend"]),
            parse_dates=["ex_date"],
        )
        if frame.empty:
            return frame
        output = frame.copy()
        output["ex_date"] = pd.to_datetime(output["ex_date"], errors="coerce")
        output["ratio"] = pd.to_numeric(output.get("ratio"), errors="coerce").fillna(1.0)
        output["cash_dividend"] = pd.to_numeric(output.get("cash_dividend"), errors="coerce").fillna(0.0)
        if symbols is not None:
            output = output[output["symbol"].isin(symbols)]
        if start is not None:
            output = output[output["ex_date"] >= pd.Timestamp(start)]
        if end is not None:
            output = output[output["ex_date"] <= pd.Timestamp(end)]
        return output.sort_values(["ex_date", "symbol"]).reset_index(drop=True)

    def get_lot_sizes(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        frame = self._load_optional_csv(
            filename="lot_size.csv",
            default=pd.DataFrame(columns=["symbol", "lot_size"]),
            parse_dates=None,
        )
        if frame.empty:
            base = self.get_security_master()[["symbol"]].copy()
            base["lot_size"] = 1
            frame = base
        output = frame.copy()
        output["lot_size"] = pd.to_numeric(output.get("lot_size"), errors="coerce").fillna(1).astype(int).clip(lower=1)
        if symbols is not None:
            output = output[output["symbol"].isin(symbols)]
        return output[["symbol", "lot_size"]].drop_duplicates("symbol").reset_index(drop=True)

    def get_industry_mapping(
        self,
        symbols: Sequence[str] | None = None,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = self._load_optional_csv(
            filename="industry_mapping.csv",
            default=pd.DataFrame(columns=["symbol", "industry_source", "industry_std", "as_of_date"]),
            parse_dates=["as_of_date"],
        )
        if frame.empty:
            master = self.get_security_master()[["symbol", "industry", "industry_std"]].copy()
            master["industry_source"] = master["industry"]
            master["as_of_date"] = pd.NaT
            output = master[["symbol", "industry_source", "industry_std", "as_of_date"]]
        else:
            output = frame.copy()
            output["as_of_date"] = pd.to_datetime(output.get("as_of_date"), errors="coerce")
            output["industry_std"] = output.get("industry_std", output.get("industry_source", "UNKNOWN"))
            output["industry_std"] = output["industry_std"].map(normalize_industry_label)
        if symbols is not None:
            output = output[output["symbol"].isin(symbols)]
        if as_of_date is not None and "as_of_date" in output.columns:
            as_of = pd.Timestamp(as_of_date)
            dated = output[output["as_of_date"].notna()].copy()
            undated = output[output["as_of_date"].isna()].copy()
            dated = dated[dated["as_of_date"] <= as_of].sort_values(["symbol", "as_of_date"]).groupby("symbol").tail(1)
            output = pd.concat([dated, undated], ignore_index=True).drop_duplicates("symbol", keep="first")
        return output[["symbol", "industry_source", "industry_std", "as_of_date"]].reset_index(drop=True)

    def _load_optional_csv(
        self,
        filename: str,
        default: pd.DataFrame,
        parse_dates: list[str] | None,
    ) -> pd.DataFrame:
        if filename in self._cache:
            return self._cache[filename].copy()
        path = self.base_dir / filename
        if not path.exists():
            self._cache[filename] = default.copy()
            return default.copy()
        frame = pd.read_csv(path)
        if parse_dates:
            for column in parse_dates:
                if column in frame.columns:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")
        self._cache[filename] = frame.copy()
        return frame
