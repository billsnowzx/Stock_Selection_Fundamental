from __future__ import annotations

import re

import pandas as pd


SECURITY_MASTER_ALIASES = {
    "ticker": "symbol",
    "code": "symbol",
    "stock_code": "symbol",
    "listing_date": "list_date",
}

PRICE_ALIASES = {
    "trade_date": "date",
    "suspend": "is_suspended",
}

FINANCIAL_ALIASES = {
    "report_date": "period_end",
    "ann_date": "release_date",
    "total_debt": "total_liabilities",
}

RELEASE_ALIASES = {
    "ann_date": "release_date",
    "report_date": "period_end",
}

STANDARD_FINANCIAL_FIELDS = (
    "symbol",
    "period_end",
    "release_date",
    "revenue",
    "net_income",
    "operating_cashflow",
    "free_cashflow",
    "invested_capital",
    "total_liabilities",
)


def ensure_datetime(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column], errors="coerce")
    return output


def rename_to_standard(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    existing = {k: v for k, v in aliases.items() if k in frame.columns and k != v}
    return frame.rename(columns=existing)


def infer_market_from_symbol(symbol: str) -> str:
    text = str(symbol).upper()
    if text.endswith(".HK") or text.startswith("^HSI"):
        return "HK"
    if text.endswith(".SH") or text.endswith(".SZ") or text.startswith("^000"):
        return "CN"
    return "UNKNOWN"


def infer_exchange_from_symbol(symbol: str) -> str:
    text = str(symbol).upper()
    if text.endswith(".HK"):
        return "HKEX"
    if text.endswith(".SH"):
        return "SSE"
    if text.endswith(".SZ"):
        return "SZSE"
    if re.match(r"^\^HS", text):
        return "HKINDEX"
    if re.match(r"^\^0", text):
        return "CNINDEX"
    return "UNKNOWN"


def standardize_security_master(frame: pd.DataFrame) -> pd.DataFrame:
    output = rename_to_standard(frame, SECURITY_MASTER_ALIASES)
    output = ensure_datetime(output, ["list_date", "delist_date"])
    if "market" not in output.columns:
        output["market"] = output["symbol"].map(infer_market_from_symbol)
    if "exchange" not in output.columns:
        output["exchange"] = output["symbol"].map(infer_exchange_from_symbol)
    return output


def standardize_price_history(frame: pd.DataFrame) -> pd.DataFrame:
    output = rename_to_standard(frame, PRICE_ALIASES)
    output = ensure_datetime(output, ["date"])
    if "is_suspended" in output.columns:
        output["is_suspended"] = output["is_suspended"].fillna(True).astype(bool)
    if "market" not in output.columns and "symbol" in output.columns:
        output["market"] = output["symbol"].map(infer_market_from_symbol)
    return output


def standardize_financials(frame: pd.DataFrame) -> pd.DataFrame:
    output = rename_to_standard(frame, FINANCIAL_ALIASES)
    output = ensure_datetime(output, ["period_end", "release_date"])
    return output


def standardize_release_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    output = rename_to_standard(frame, RELEASE_ALIASES)
    output = ensure_datetime(output, ["period_end", "release_date"])
    return output
