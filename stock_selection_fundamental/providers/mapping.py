from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class FieldMappingVersion:
    source: str
    version: str
    security_master: dict[str, str]
    price: dict[str, str]
    financial: dict[str, str]
    release_calendar: dict[str, str]


MAPPING_REGISTRY: dict[str, FieldMappingVersion] = {
    "hk_akshare_v1": FieldMappingVersion(
        source="hk_akshare",
        version="v1",
        security_master={"ticker": "symbol", "stock_code": "symbol", "listing_date": "list_date"},
        price={"trade_date": "date", "suspend": "is_suspended"},
        financial={"report_date": "period_end", "ann_date": "release_date", "total_debt": "total_liabilities"},
        release_calendar={"ann_date": "release_date", "report_date": "period_end"},
    ),
    "cn_akshare_v1": FieldMappingVersion(
        source="cn_akshare",
        version="v1",
        security_master={"ticker": "symbol", "stock_code": "symbol", "listing_date": "list_date"},
        price={"trade_date": "date", "suspend": "is_suspended"},
        financial={"report_date": "period_end", "ann_date": "release_date", "total_debt": "total_liabilities"},
        release_calendar={"ann_date": "release_date", "report_date": "period_end"},
    ),
    "local_csv_v1": FieldMappingVersion(
        source="local_csv",
        version="v1",
        security_master={"ticker": "symbol", "code": "symbol", "stock_code": "symbol", "listing_date": "list_date"},
        price={"trade_date": "date", "suspend": "is_suspended"},
        financial={"report_date": "period_end", "ann_date": "release_date", "total_debt": "total_liabilities"},
        release_calendar={"ann_date": "release_date", "report_date": "period_end"},
    ),
}


INDUSTRY_CANONICAL_MAP: dict[str, str] = {
    "technology": "TECH",
    "information technology": "TECH",
    "consumer": "CONSUMER",
    "consumer discretionary": "CONSUMER",
    "consumer staples": "CONSUMER",
    "healthcare": "HEALTHCARE",
    "medical": "HEALTHCARE",
    "industrials": "INDUSTRIALS",
    "materials": "MATERIALS",
    "utilities": "UTILITIES",
    "financials": "FINANCIALS",
    "real estate": "REAL_ESTATE",
    "telecommunications": "TELECOM",
    "communication services": "TELECOM",
    "energy": "ENERGY",
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


def resolve_mapping(mapping_key: str | None) -> FieldMappingVersion:
    if not mapping_key:
        return MAPPING_REGISTRY["local_csv_v1"]
    if mapping_key not in MAPPING_REGISTRY:
        raise KeyError(f"Unknown mapping key: {mapping_key}")
    return MAPPING_REGISTRY[mapping_key]


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


def normalize_industry_label(label: str | None) -> str:
    if label is None or pd.isna(label):
        return "UNKNOWN"
    key = str(label).strip().lower()
    return INDUSTRY_CANONICAL_MAP.get(key, key.upper().replace(" ", "_"))


def standardize_security_master(frame: pd.DataFrame, mapping_key: str = "local_csv_v1") -> pd.DataFrame:
    mapping = resolve_mapping(mapping_key)
    output = rename_to_standard(frame, mapping.security_master)
    output = ensure_datetime(output, ["list_date", "delist_date"])
    if "market" not in output.columns:
        output["market"] = output["symbol"].map(infer_market_from_symbol)
    if "exchange" not in output.columns:
        output["exchange"] = output["symbol"].map(infer_exchange_from_symbol)
    if "industry" in output.columns:
        output["industry_std"] = output["industry"].map(normalize_industry_label)
    else:
        output["industry_std"] = "UNKNOWN"
    return output


def standardize_price_history(frame: pd.DataFrame, mapping_key: str = "local_csv_v1") -> pd.DataFrame:
    mapping = resolve_mapping(mapping_key)
    output = rename_to_standard(frame, mapping.price)
    output = ensure_datetime(output, ["date"])
    if "is_suspended" in output.columns:
        output["is_suspended"] = output["is_suspended"].fillna(True).astype(bool)
    if "market" not in output.columns and "symbol" in output.columns:
        output["market"] = output["symbol"].map(infer_market_from_symbol)
    return output


def standardize_financials(frame: pd.DataFrame, mapping_key: str = "local_csv_v1") -> pd.DataFrame:
    mapping = resolve_mapping(mapping_key)
    output = rename_to_standard(frame, mapping.financial)
    output = ensure_datetime(output, ["period_end", "release_date"])
    return output


def standardize_release_calendar(frame: pd.DataFrame, mapping_key: str = "local_csv_v1") -> pd.DataFrame:
    mapping = resolve_mapping(mapping_key)
    output = rename_to_standard(frame, mapping.release_calendar)
    output = ensure_datetime(output, ["period_end", "release_date"])
    return output
