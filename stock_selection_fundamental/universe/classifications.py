from __future__ import annotations

import pandas as pd


def map_industry(universe: pd.DataFrame, mapping: pd.DataFrame | None = None) -> pd.DataFrame:
    output = universe.copy()
    if mapping is None or mapping.empty or "industry" in output.columns:
        return output
    if {"symbol", "industry"}.issubset(mapping.columns):
        output = output.merge(mapping[["symbol", "industry"]], on="symbol", how="left")
    return output
