from __future__ import annotations

import pandas as pd

from .eligibility import filter_delisted, filter_listing_age, filter_security_types


def _filter_liquidity(
    universe: pd.DataFrame,
    price_history: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int,
    min_avg_turnover: float,
) -> pd.DataFrame:
    if universe.empty:
        return universe
    history = price_history[price_history["date"] <= as_of].copy()
    if history.empty:
        return universe.head(0)
    history = history.sort_values(["symbol", "date"]) 
    recent = history.groupby("symbol", group_keys=False).tail(lookback_days)
    avg_turnover = recent.groupby("symbol")["turnover"].mean().rename("avg_turnover_lookback")
    merged = universe.merge(avg_turnover, on="symbol", how="left")
    return merged[merged["avg_turnover_lookback"].fillna(0) >= min_avg_turnover]


def _filter_special_treatment(universe: pd.DataFrame, enable_st_filter: bool) -> pd.DataFrame:
    if not enable_st_filter:
        return universe
    output = universe.copy()
    if "name" in output.columns:
        output = output[~output["name"].astype(str).str.contains("ST", case=False, na=False)]
    return output


def _filter_industry(universe: pd.DataFrame, config: dict) -> pd.DataFrame:
    output = universe.copy()
    if "industry" not in output.columns:
        return output
    include = config.get("include_industries")
    exclude = config.get("exclude_industries")
    if include:
        output = output[output["industry"].isin(include)]
    if exclude:
        output = output[~output["industry"].isin(exclude)]
    return output


def build_universe(
    master: pd.DataFrame,
    price_history: pd.DataFrame,
    trading_status: pd.DataFrame,
    as_of: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    output = master.copy()
    output = output[output["board"] == config.get("board", "MAIN")]
    output = filter_security_types(
        output,
        include_type=config.get("security_type", "EQUITY"),
        excluded=tuple(config.get("excluded_security_types", [])),
    )
    output = filter_delisted(output, as_of)
    output = filter_listing_age(output, as_of, int(config.get("min_listing_days", 252)))

    if output.empty:
        return output

    status_cols = ["symbol", "is_tradable", "is_suspended", "open", "close", "turnover"]
    status = trading_status[[col for col in status_cols if col in trading_status.columns]].copy()
    output = output.merge(status, on="symbol", how="left")
    output = output[output["is_tradable"].fillna(False)]
    output = _filter_special_treatment(output, bool(config.get("exclude_st", False)))
    output = _filter_industry(output, config)
    output = _filter_liquidity(
        output,
        price_history=price_history,
        as_of=as_of,
        lookback_days=int(config.get("liquidity_lookback_days", 20)),
        min_avg_turnover=float(config.get("min_avg_turnover", 0.0)),
    )
    output = output[output["close"].notna()]
    return output.sort_values("symbol").reset_index(drop=True)
