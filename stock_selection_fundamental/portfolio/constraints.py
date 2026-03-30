from __future__ import annotations

import pandas as pd


def enforce_weight_limits(weights: pd.Series, max_weight: float) -> pd.Series:
    output = weights.copy().clip(lower=0, upper=max_weight)
    total = output.sum()
    return output / total if total > 0 else output


def enforce_holding_count(weights: pd.Series, min_holdings: int, max_holdings: int | None) -> pd.Series:
    output = weights[weights > 0].copy()
    if max_holdings is not None and len(output) > max_holdings:
        output = output.sort_values(ascending=False).head(max_holdings)
    if len(output) < min_holdings:
        return pd.Series(dtype=float)
    total = output.sum()
    return output / total if total > 0 else output
