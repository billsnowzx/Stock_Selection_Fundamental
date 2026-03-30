from __future__ import annotations

import pandas as pd


def neutralize_placeholder(weights: pd.Series) -> pd.Series:
    if weights.empty:
        return weights
    total = weights.sum()
    return weights / total if total > 0 else weights
