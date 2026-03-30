from __future__ import annotations

import pandas as pd


def portfolio_turnover(prev: pd.Series, curr: pd.Series) -> float:
    all_symbols = prev.index.union(curr.index)
    prev_aligned = prev.reindex(all_symbols).fillna(0.0)
    curr_aligned = curr.reindex(all_symbols).fillna(0.0)
    return float((curr_aligned - prev_aligned).abs().sum() / 2)
