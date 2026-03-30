from __future__ import annotations

import pandas as pd

from .constraints import enforce_holding_count, enforce_weight_limits


def build_target_weights(
    selected: pd.DataFrame,
    method: str,
    max_weight: float,
    min_holdings: int,
    max_holdings: int | None,
) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)

    if method == "score_weight":
        raw = selected.set_index("symbol")["composite_score"].clip(lower=0)
        if raw.sum() == 0:
            raw = pd.Series(1.0, index=raw.index)
        weights = raw / raw.sum()
    else:
        weights = pd.Series(1.0 / len(selected), index=selected["symbol"])

    weights = enforce_weight_limits(weights, max_weight=max_weight)
    weights = enforce_holding_count(weights, min_holdings=min_holdings, max_holdings=max_holdings)
    return weights
