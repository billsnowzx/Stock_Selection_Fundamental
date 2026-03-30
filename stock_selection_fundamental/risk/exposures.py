from __future__ import annotations

import pandas as pd


def estimate_basic_exposures(selection_history: pd.DataFrame) -> pd.DataFrame:
    if selection_history.empty:
        return pd.DataFrame(columns=["signal_date", "industry", "weight"])
    if "industry" not in selection_history.columns:
        return pd.DataFrame(columns=["signal_date", "industry", "weight"])
    grouped = (
        selection_history.groupby(["signal_date", "industry"])["target_weight"]
        .sum()
        .reset_index(name="weight")
    )
    return grouped


def estimate_style_exposure(candidates: pd.DataFrame) -> pd.DataFrame:
    """Heuristic style proxy to support soft constraints before full risk model integration."""
    if candidates.empty or "symbol" not in candidates.columns:
        return pd.DataFrame(columns=["symbol", "size", "value", "momentum"])
    frame = candidates.copy()
    frame["size"] = (
        pd.to_numeric(frame.get("avg_turnover_lookback"), errors="coerce")
        .rank(pct=True)
        .fillna(0.5)
        - 0.5
    )
    frame["value"] = (
        pd.to_numeric(frame.get("net_margin"), errors="coerce")
        .rank(pct=True)
        .fillna(0.5)
        - 0.5
    )
    frame["momentum"] = (
        pd.to_numeric(frame.get("revenue_growth_yoy"), errors="coerce")
        .rank(pct=True)
        .fillna(0.5)
        - 0.5
    )
    return frame[["symbol", "size", "value", "momentum"]]
