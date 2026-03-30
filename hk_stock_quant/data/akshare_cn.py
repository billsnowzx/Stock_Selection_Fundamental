from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .local_csv import LocalCSVDataProvider

try:
    import akshare as ak
except ImportError:  # pragma: no cover - optional dependency at runtime
    ak = None


class AkshareCNDataProvider(LocalCSVDataProvider):
    REPORT_LAG_DAYS = {
        "年报": 90,
        "中报": 60,
        "一季报": 45,
        "三季报": 45,
    }
    MAX_RETRIES = 3
    RETRY_BASE_SLEEP_SECONDS = 1.0

    @classmethod
    def _call_with_retry(cls, func, *args, **kwargs):
        last_error: Exception | None = None
        for attempt in range(1, cls.MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - network resilience
                last_error = exc
                if attempt >= cls.MAX_RETRIES:
                    raise
                time.sleep(cls.RETRY_BASE_SLEEP_SECONDS * attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected retry state")

    @classmethod
    def sync_to_local_dataset(
        cls,
        output_dir: str | Path,
        start: str,
        end: str,
        symbols: list[str] | None = None,
        max_symbols: int | None = 100,
        benchmark_symbol: str = "sh000300",
        sleep_seconds: float = 0.1,
    ) -> Path:
        if ak is None:
            raise ImportError("akshare is required to sync an A-share dataset.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        security_master = cls._fetch_security_master(symbols=symbols, max_symbols=max_symbols)
        selected_symbols = security_master["symbol"].tolist()

        price_frames: list[pd.DataFrame] = []
        financial_frames: list[pd.DataFrame] = []
        release_frames: list[pd.DataFrame] = []
        listing_dates: dict[str, pd.Timestamp] = {}

        for idx, symbol in enumerate(selected_symbols, start=1):
            print(f"[{idx}/{len(selected_symbols)}] syncing {symbol}")

            try:
                price_frame = cls._fetch_price_history(symbol=symbol, start=start, end=end)
            except Exception as exc:  # pragma: no cover - network resilience
                print(f"[WARN] price sync failed for {symbol}: {exc}")
                price_frame = pd.DataFrame()
            if not price_frame.empty:
                price_frames.append(price_frame)
                listing_dates[symbol] = pd.Timestamp(price_frame["date"].min())

            try:
                financial_frame, release_frame = cls._fetch_financial_bundle(symbol=symbol)
            except Exception as exc:  # pragma: no cover - network resilience
                print(f"[WARN] financial sync failed for {symbol}: {exc}")
                financial_frame = pd.DataFrame()
                release_frame = pd.DataFrame()
            if not financial_frame.empty:
                financial_frames.append(financial_frame)
                if not release_frame.empty:
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
        spot = cls._call_with_retry(ak.stock_zh_a_spot_em)
        codes = spot["代码"].astype(str).str.zfill(6)
        frame = pd.DataFrame(
            {
                "symbol": codes.map(cls._normalize_symbol),
                "name": spot["名称"].astype(str),
                "board": "MAIN",
                "security_type": "EQUITY",
                "industry": pd.NA,
            }
        )
        frame = frame[frame["symbol"].notna()].reset_index(drop=True)

        if symbols:
            normalized = {cls._normalize_symbol(symbol) for symbol in symbols}
            normalized = {value for value in normalized if value is not None}
            filtered = frame[frame["symbol"].isin(normalized)].copy()
            missing = sorted(normalized - set(filtered["symbol"]))
            if missing:
                missing_rows = pd.DataFrame(
                    {
                        "symbol": missing,
                        "name": [f"UNKNOWN_{item}" for item in missing],
                        "board": "MAIN",
                        "security_type": "EQUITY",
                        "industry": pd.NA,
                    }
                )
                filtered = pd.concat([filtered, missing_rows], ignore_index=True)
            frame = filtered
        if max_symbols is not None:
            frame = frame.head(max_symbols)

        frame["list_date"] = pd.NaT
        frame["delist_date"] = pd.NaT
        return frame.reset_index(drop=True)

    @classmethod
    def _fetch_price_history(cls, symbol: str, start: str, end: str) -> pd.DataFrame:
        code = cls._strip_market_suffix(symbol)
        frame = cls._call_with_retry(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=pd.Timestamp(start).strftime("%Y%m%d"),
            end_date=pd.Timestamp(end).strftime("%Y%m%d"),
            adjust="",
        )
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "turnover",
                    "is_suspended",
                ]
            )

        normalized = pd.DataFrame(
            {
                "date": pd.to_datetime(frame["日期"]),
                "symbol": cls._normalize_symbol(code),
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
        api_symbol = cls._normalize_index_symbol(symbol)
        frame = cls._call_with_retry(ak.stock_zh_index_daily, symbol=api_symbol)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))].copy()
        normalized_symbol = f"^{api_symbol[-6:]}"
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
        em_symbol = cls._to_em_symbol(symbol)
        sina_symbol = cls._to_sina_symbol(symbol)

        analysis = cls._call_with_retry(
            ak.stock_financial_analysis_indicator_em,
            symbol=em_symbol,
            indicator="按报告期",
        )
        if analysis.empty:
            empty_financials = pd.DataFrame(columns=cls._financial_columns())
            empty_releases = pd.DataFrame(columns=["symbol", "period_end", "release_date"])
            return empty_financials, empty_releases

        analysis = analysis.copy()
        analysis["period_end"] = pd.to_datetime(analysis["REPORT_DATE"])
        analysis["release_date"] = pd.to_datetime(analysis.get("NOTICE_DATE"), errors="coerce")

        balance = cls._call_with_retry(ak.stock_financial_report_sina, stock=sina_symbol, symbol="资产负债表")
        cashflow = cls._call_with_retry(ak.stock_financial_report_sina, stock=sina_symbol, symbol="现金流量表")

        liabilities = cls._extract_statement_values(
            balance,
            {
                "total_liabilities": ["负债合计"],
                "current_liabilities": ["流动负债合计"],
                "total_assets": ["资产总计"],
            },
        )
        cash_items = cls._extract_statement_values(
            cashflow,
            {
                "operating_cashflow": ["经营活动产生的现金流量净额"],
            },
        )
        capex = cls._extract_statement_sum(
            cashflow,
            "capital_expenditure",
            [
                "购建固定资产、无形资产和其他长期资产所支付的现金",
                "购建固定资产、无形资产和其他长期资产支付的现金",
            ],
        )

        frame = analysis.merge(liabilities, on="period_end", how="left")
        frame = frame.merge(cash_items, on="period_end", how="left")
        frame = frame.merge(capex, on="period_end", how="left")

        frame["symbol"] = cls._normalize_symbol(symbol)
        frame["report_type"] = frame.get("REPORT_DATE_NAME", "REPORT").fillna("REPORT")
        frame["revenue"] = pd.to_numeric(frame.get("TOTALOPERATEREVE"), errors="coerce")
        frame["net_income"] = pd.to_numeric(frame.get("PARENTNETPROFIT"), errors="coerce")
        frame["effective_tax_rate"] = pd.to_numeric(frame.get("TAXRATE"), errors="coerce") / 100.0
        frame["roic"] = pd.to_numeric(frame.get("ROIC"), errors="coerce") / 100.0
        frame["net_margin"] = pd.to_numeric(frame.get("XSJLL"), errors="coerce") / 100.0
        frame["revenue_growth_yoy"] = pd.to_numeric(frame.get("TOTALOPERATEREVETZ"), errors="coerce") / 100.0
        frame["net_income_growth_yoy"] = pd.to_numeric(frame.get("PARENTNETPROFITTZ"), errors="coerce") / 100.0

        frame["total_liabilities"] = pd.to_numeric(frame.get("total_liabilities"), errors="coerce")
        frame["current_liabilities"] = pd.to_numeric(frame.get("current_liabilities"), errors="coerce")
        frame["total_assets"] = pd.to_numeric(frame.get("total_assets"), errors="coerce")
        frame["invested_capital"] = frame["total_assets"] - frame["current_liabilities"]

        frame["operating_cashflow"] = pd.to_numeric(frame.get("operating_cashflow"), errors="coerce")
        frame["capital_expenditure"] = pd.to_numeric(frame.get("capital_expenditure"), errors="coerce").abs()
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

        release_date = pd.to_datetime(frame["release_date"], errors="coerce")
        missing = release_date.isna()
        if missing.any():
            fallback = frame.loc[missing].apply(
                lambda row: cls._estimate_release_date(row["period_end"], str(row.get("report_type", ""))),
                axis=1,
            )
            release_date.loc[missing] = pd.to_datetime(fallback, errors="coerce")

        releases = pd.DataFrame(
            {
                "symbol": frame["symbol"],
                "period_end": frame["period_end"],
                "release_date": pd.to_datetime(release_date),
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

        result = pd.DataFrame({"period_end": pd.to_datetime(statement_frame["报告日"], errors="coerce")})
        for target_column, candidates in mapping.items():
            result[target_column] = np.nan
            for candidate in candidates:
                if candidate in statement_frame.columns:
                    values = pd.to_numeric(statement_frame[candidate], errors="coerce")
                    result[target_column] = result[target_column].fillna(values)
        return result.dropna(subset=["period_end"]).reset_index(drop=True)

    @classmethod
    def _extract_statement_sum(
        cls,
        statement_frame: pd.DataFrame,
        target_column: str,
        candidates: list[str],
    ) -> pd.DataFrame:
        if statement_frame.empty:
            return pd.DataFrame(columns=["period_end", target_column])

        result = pd.DataFrame({"period_end": pd.to_datetime(statement_frame["报告日"], errors="coerce")})
        matched = [name for name in candidates if name in statement_frame.columns]
        if matched:
            result[target_column] = (
                statement_frame[matched].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            )
        else:
            result[target_column] = np.nan
        return result.dropna(subset=["period_end"]).reset_index(drop=True)

    @classmethod
    def _estimate_release_date(cls, period_end: pd.Timestamp, report_type: str) -> pd.Timestamp:
        for key, lag_days in cls.REPORT_LAG_DAYS.items():
            if key in report_type:
                return pd.Timestamp(period_end) + pd.Timedelta(days=lag_days)
        return pd.Timestamp(period_end) + pd.Timedelta(days=60)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str | None:
        digits = re.sub(r"\D", "", str(symbol))
        if len(digits) < 6:
            return None
        code = digits[-6:]
        if code.startswith("6"):
            return f"{code}.SH"
        if code.startswith(("0", "3")):
            return f"{code}.SZ"
        return None

    @staticmethod
    def _strip_market_suffix(symbol: str) -> str:
        digits = re.sub(r"\D", "", str(symbol))
        return digits[-6:]

    @classmethod
    def _to_em_symbol(cls, symbol: str) -> str:
        normalized = cls._normalize_symbol(symbol)
        if normalized is None:
            raise ValueError(f"Unsupported A-share symbol: {symbol}")
        return normalized

    @classmethod
    def _to_sina_symbol(cls, symbol: str) -> str:
        code = cls._strip_market_suffix(symbol)
        return f"sh{code}" if code.startswith("6") else f"sz{code}"

    @staticmethod
    def _normalize_index_symbol(symbol: str) -> str:
        text = str(symbol).lower()
        if text.startswith(("sh", "sz")) and len(text) >= 8:
            return text
        digits = re.sub(r"\D", "", text)
        if len(digits) != 6:
            return "sh000300"
        return f"sh{digits}" if digits.startswith("0") else f"sz{digits}"

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

