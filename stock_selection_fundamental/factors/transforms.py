from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, lower: float, upper: float) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series
    return series.clip(valid.quantile(lower), valid.quantile(upper))


def zscore(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index).where(series.notna())
    return ((series - valid.mean()) / std).where(series.notna())


def rank_normalize(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    ranks = series.rank(method="average", pct=True)
    return (ranks - 0.5).where(series.notna())
