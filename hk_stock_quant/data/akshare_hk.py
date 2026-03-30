from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .local_csv import LocalCSVDataProvider

try:
    import akshare as ak
except ImportError:  # pragma: no cover - optional dependency at runtime
    ak = None


class AkshareHKDataProvider(LocalCSVDataProvider):
    RELEASE_LAG_DAYS = {
        "001": 90,
        "002": 60,
        "003": 45,
        "004": 45,
    }

    @classmethod
    def sync_to_local_dataset(
        cls,
        output_dir: str | Path,
        start: str,
        end: str,
        symbols: list[str] | None = None,
        max_symbols: int | None = 50,
        benchmark_symbol: str = "HSI",
        sleep_seconds: float = 0.2,
    ) -> Path:
        if ak is None:
            raise ImportError("akshare is required to sync a real Hong Kong dataset.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        security_master = cls._fetch_security_master(symbols=symbols, max_symbols=max_symbols)
        selected_symbols = security_master["symbol"].tolist()

        price_frames: list[pd.DataFrame] = []
        financial_frames: list[pd.DataFrame] = []
        release_frames: list[pd.DataFrame] = []
        listing_dates: dict[str, pd.Timestamp] = {}

        for idx, symbol in enumerate(selected_symbols, start=1):
            code = cls._strip_market_suffix(symbol)
            print(f"[{idx}/{len(selected_symbols)}] syncing {symbol}")

            price_frame = cls._fetch_price_history(symbol=code, start=start, end=end)
            if not price_frame.empty:
                price_frames.append(price_frame)
                listing_dates[symbol] = pd.Timestamp(price_frame["date"].min())

            financial_frame, release_frame = cls._fetch_financial_bundle(symbol=code)
            if not financial_frame.empty:
                financial_frames.append(financial_frame)
                release_frames.append(release_frame)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        security_master["list_date"] = security_master["symbol"].map(listing_dates)
        security_master["delist_date"] = pd.NaT

        benchmark_frame = cls._fetch_benchmark_history(symbol=benchmark_symbol, start=start, end=end)
        all_prices = pd.concat(price_frames + [benchmark_frame], ignore_index=True) if price_frames else benchmark_frame
        all_prices = all_prices.sort_values(["date", "symbol"]).reset_index(drop=True)

        all_financials = (
            pd.concat(financial_frames, ignore_index=True).sort_values(["symbol", "period_end"])
            if financial_frames
            else pd.DataFrame(columns=cls._financial_columns())
        )
        all_releases = (
            pd.concat(release_frames, ignore_index=True).drop_duplicates(["symbol", "period_end"])
            if release_frames
            else pd.DataFrame(columns=["symbol", "period_end", "release_date"])
        )

        security_master.to_csv(output_path / "security_master.csv", index=False)
        all_prices.to_csv(output_path / "price_history.csv", index=False)
        all_financials.to_csv(output_path / "financials.csv", index=False)
        all_releases.to_csv(output_path / "release_calendar.csv", index=False)
        return output_path

    @classmethod
    def _fetch_security_master(
        cls,
        symbols: list[str] | None = None,
        max_symbols: int | None = None,
    ) -> pd.DataFrame:
        spot = ak.stock_hk_main_board_spot_em()
        frame = pd.DataFrame(
            {
                "symbol": spot["代码"].astype(str).str.zfill(5) + ".HK",
                "name": spot["名称"].astype(str),
                "board": "MAIN",
                "security_type": "EQUITY",
                "industry": pd.NA,
            }
        )
        if symbols:
            normalized = {cls._normalize_symbol(symbol) for symbol in symbols}
            frame = frame[frame["symbol"].isin(normalized)]
        if max_symbols is not None:
            frame = frame.head(max_symbols)
        frame["list_date"] = pd.NaT
        frame["delist_date"] = pd.NaT
        return frame.reset_index(drop=True)

    @classmethod
    def _fetch_price_history(cls, symbol: str, start: str, end: str) -> pd.DataFrame:
        frame = ak.stock_hk_hist(
            symbol=symbol,
            period="daily",
            start_date=pd.Timestamp(start).strftime("%Y%m%d"),
            end_date=pd.Timestamp(end).strftime("%Y%m%d"),
            adjust="",
        )
        if frame.empty:
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "turnover", "is_suspended"])
        normalized = pd.DataFrame(
            {
                "date": pd.to_datetime(frame["日期"]),
                "symbol": cls._normalize_symbol(symbol),
                "open": pd.to_numeric(frame["开盘"], errors="coerce"),
                "high": pd.to_numeric(frame["最高"], errors="coerce"),
                "low": pd.to_numeric(frame["最低"], errors="coerce"),
                "close": pd.to_numeric(frame["收盘"], errors="coerce"),
                "volume": pd.to_numeric(frame["成交量"], errors="coerce"),
                "turnover": pd.to_numeric(frame["成交额"], errors="coerce"),
                "is_suspended": False,
            }
        )
        return normalized.dropna(subset=["date", "close"]).reset_index(drop=True)

    @classmethod
    def _fetch_benchmark_history(cls, symbol: str, start: str, end: str) -> pd.DataFrame:
        frame = ak.stock_hk_index_daily_sina(symbol=symbol)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].copy()
        normalized_symbol = f"^{symbol.upper()}"
        return pd.DataFrame(
            {
                "date": frame["date"],
                "symbol": normalized_symbol,
                "open": pd.to_numeric(frame["open"], errors="coerce"),
                "high": pd.to_numeric(frame["high"], errors="coerce"),
                "low": pd.to_numeric(frame["low"], errors="coerce"),
                "close": pd.to_numeric(frame["close"], errors="coerce"),
                "volume": pd.to_numeric(frame["volume"], errors="coerce"),
                "turnover": np.nan,
                "is_suspended": False,
            }
        ).dropna(subset=["date", "close"]).reset_index(drop=True)

    @classmethod
    def _fetch_financial_bundle(cls, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        analysis = ak.stock_financial_hk_analysis_indicator_em(symbol=symbol, indicator="报告期")
        if analysis.empty:
            empty_financials = pd.DataFrame(columns=cls._financial_columns())
            empty_releases = pd.DataFrame(columns=["symbol", "period_end", "release_date"])
            return empty_financials, empty_releases

        analysis = analysis.copy()
        analysis["REPORT_DATE"] = pd.to_datetime(analysis["REPORT_DATE"])
        analysis["period_end"] = analysis["REPORT_DATE"]

        balance = ak.stock_financial_hk_report_em(stock=symbol, symbol="资产负债表", indicator="报告期")
        cashflow = ak.stock_financial_hk_report_em(stock=symbol, symbol="现金流量表", indicator="报告期")

        liabilities = cls._extract_statement_values(
            balance,
            {
                "total_liabilities": ["总负债"],
                "invested_capital": ["总资产减流动负债", "总资产减总负债合计"],
            },
        )
        cash_items = cls._extract_statement_values(cashflow, {"operating_cashflow": ["经营业务现金净额"]})
        capex = cls._extract_statement_sum(cashflow, "capital_expenditure", ["购建固定资产", "购建无形资产及其他资产"])

        frame = analysis.merge(liabilities, on="period_end", how="left")
        frame = frame.merge(cash_items, on="period_end", how="left")
        frame = frame.merge(capex, on="period_end", how="left")

        frame["symbol"] = cls._normalize_symbol(symbol)
        frame["report_type"] = frame["DATE_TYPE_CODE"].astype(str).map(
            {"001": "ANNUAL", "002": "INTERIM", "003": "Q1", "004": "Q3"}
        ).fillna("REPORT")
        frame["revenue"] = pd.to_numeric(frame["OPERATE_INCOME"], errors="coerce")
        frame["net_income"] = pd.to_numeric(frame["HOLDER_PROFIT"], errors="coerce")
        frame["effective_tax_rate"] = pd.to_numeric(frame["TAX_EBT"], errors="coerce") / 100.0
        frame["roic"] = pd.to_numeric(frame["ROIC_YEARLY"], errors="coerce") / 100.0
        frame["net_margin"] = pd.to_numeric(frame["NET_PROFIT_RATIO"], errors="coerce") / 100.0
        frame["revenue_growth_yoy"] = pd.to_numeric(frame["OPERATE_INCOME_YOY"], errors="coerce") / 100.0
        frame["net_income_growth_yoy"] = pd.to_numeric(frame["HOLDER_PROFIT_YOY"], errors="coerce") / 100.0
        frame["capital_expenditure"] = pd.to_numeric(frame["capital_expenditure"], errors="coerce").abs()
        frame["operating_cashflow"] = pd.to_numeric(frame["operating_cashflow"], errors="coerce")
        frame["total_liabilities"] = pd.to_numeric(frame["total_liabilities"], errors="coerce")
        frame["invested_capital"] = pd.to_numeric(frame["invested_capital"], errors="coerce")
        frame["free_cashflow"] = frame["operating_cashflow"] - frame["capital_expenditure"]
        frame["fcf_conversion"] = frame["free_cashflow"] / frame["net_income"]
        frame["debt_to_cashflow"] = frame["total_liabilities"] / frame["operating_cashflow"]
        frame["ebit"] = np.nan
        frame["nopat"] = np.nan

        financials = frame[
            [
                "symbol",
                "period_end",
                "report_type",
                "revenue",
                "net_income",
                "ebit",
                "effective_tax_rate",
                "invested_capital",
                "total_liabilities",
                "operating_cashflow",
                "capital_expenditure",
                "free_cashflow",
                "nopat",
                "roic",
                "net_margin",
                "revenue_growth_yoy",
                "net_income_growth_yoy",
                "fcf_conversion",
                "debt_to_cashflow",
            ]
        ].copy()
        financials = financials.sort_values("period_end").reset_index(drop=True)

        releases = pd.DataFrame(
            {
                "symbol": frame["symbol"],
                "period_end": frame["period_end"],
                "release_date": frame.apply(
                    lambda row: cls._estimate_release_date(row["period_end"], row.get("DATE_TYPE_CODE")),
                    axis=1,
                ),
            }
        )
        return financials, releases

    @classmethod
    def _extract_statement_values(
        cls,
        statement_frame: pd.DataFrame,
        mapping: dict[str, list[str]],
    ) -> pd.DataFrame:
        if statement_frame.empty:
            return pd.DataFrame(columns=["period_end", *mapping.keys()])
        pivot = cls._pivot_statement_items(statement_frame)
        result = pd.DataFrame(index=pivot.index)
        for target_column, item_names in mapping.items():
            series = pd.Series(np.nan, index=pivot.index, dtype=float)
            for item_name in item_names:
                if item_name in pivot.columns:
                    series = series.fillna(pivot[item_name])
            result[target_column] = series
        result = result.reset_index().rename(columns={"REPORT_DATE": "period_end"})
        return result

    @classmethod
    def _extract_statement_sum(
        cls,
        statement_frame: pd.DataFrame,
        target_column: str,
        item_names: list[str],
    ) -> pd.DataFrame:
        if statement_frame.empty:
            return pd.DataFrame(columns=["period_end", target_column])
        pivot = cls._pivot_statement_items(statement_frame)
        existing = [name for name in item_names if name in pivot.columns]
        result = pd.DataFrame(index=pivot.index)
        if existing:
            result[target_column] = pivot[existing].sum(axis=1)
        else:
            result[target_column] = np.nan
        result = result.reset_index().rename(columns={"REPORT_DATE": "period_end"})
        return result

    @classmethod
    def _pivot_statement_items(cls, statement_frame: pd.DataFrame) -> pd.DataFrame:
        frame = statement_frame[["REPORT_DATE", "STD_ITEM_NAME", "AMOUNT"]].copy()
        frame["REPORT_DATE"] = pd.to_datetime(frame["REPORT_DATE"])
        frame["AMOUNT"] = pd.to_numeric(frame["AMOUNT"], errors="coerce")
        return frame.pivot_table(index="REPORT_DATE", columns="STD_ITEM_NAME", values="AMOUNT", aggfunc="last")

    @classmethod
    def _estimate_release_date(cls, period_end: pd.Timestamp, date_type_code: object) -> pd.Timestamp:
        code = str(date_type_code).zfill(3)
        lag_days = cls.RELEASE_LAG_DAYS.get(code, 60)
        return pd.Timestamp(period_end) + pd.Timedelta(days=lag_days)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        text = str(symbol).upper().replace(".HK", "")
        return f"{text.zfill(5)}.HK"

    @staticmethod
    def _strip_market_suffix(symbol: str) -> str:
        return str(symbol).upper().replace(".HK", "")

    @staticmethod
    def _financial_columns() -> list[str]:
        return [
            "symbol",
            "period_end",
            "report_type",
            "revenue",
            "net_income",
            "ebit",
            "effective_tax_rate",
            "invested_capital",
            "total_liabilities",
            "operating_cashflow",
            "capital_expenditure",
            "free_cashflow",
            "nopat",
            "roic",
            "net_margin",
            "revenue_growth_yoy",
            "net_income_growth_yoy",
            "fcf_conversion",
            "debt_to_cashflow",
        ]

