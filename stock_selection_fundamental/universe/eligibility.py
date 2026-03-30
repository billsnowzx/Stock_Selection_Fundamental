from __future__ import annotations

import pandas as pd


def filter_listing_age(master: pd.DataFrame, as_of: pd.Timestamp, min_days: int) -> pd.DataFrame:
    output = master.copy()
    output["list_date"] = pd.to_datetime(output["list_date"], errors="coerce")
    output = output[output["list_date"].notna() & (output["list_date"] <= as_of)]
    age = (as_of - output["list_date"]).dt.days
    output = output[age >= min_days]
    return output


def filter_security_types(master: pd.DataFrame, include_type: str, excluded: tuple[str, ...]) -> pd.DataFrame:
    output = master.copy()
    if "security_type" not in output.columns:
        return output.head(0)
    output = output[output["security_type"] == include_type]
    if excluded:
        output = output[~output["security_type"].isin(excluded)]
    return output


def filter_delisted(master: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    output = master.copy()
    output["delist_date"] = pd.to_datetime(output["delist_date"], errors="coerce")
    return output[output["delist_date"].isna() | (output["delist_date"] >= as_of)]
