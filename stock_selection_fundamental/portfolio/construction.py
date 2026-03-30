from __future__ import annotations

import pandas as pd

from .constraints import (
    enforce_holding_count,
    enforce_liquidity_constraint,
    enforce_non_tradable_filter,
    enforce_style_soft_constraints,
    enforce_weight_limits,
)


def build_target_weights(
    selected: pd.DataFrame,
    method: str,
    max_weight: float,
    min_holdings: int,
    max_holdings: int | None,
    tradable_symbols: set[str] | None = None,
    portfolio_value: float | None = None,
    max_adv_participation: float | None = None,
    style_exposure: pd.DataFrame | None = None,
    style_limits: dict[str, float] | None = None,
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

    if tradable_symbols is not None:
        weights = enforce_non_tradable_filter(weights, tradable_symbols=tradable_symbols)
    if weights.empty:
        return weights

    weights = enforce_weight_limits(weights, max_weight=max_weight)
    weights = enforce_holding_count(weights, min_holdings=min_holdings, max_holdings=max_holdings)
    if weights.empty:
        return weights

    if max_adv_participation is not None and portfolio_value is not None and "avg_turnover_lookback" in selected.columns:
        avg_turnover = selected.set_index("symbol")["avg_turnover_lookback"]
        weights = enforce_liquidity_constraint(
            weights=weights,
            avg_turnover=avg_turnover,
            portfolio_value=float(portfolio_value),
            max_adv_participation=float(max_adv_participation),
        )

    weights = enforce_style_soft_constraints(
        weights=weights,
        style_exposure=style_exposure,
        style_limits=style_limits,
    )
    total = weights.sum()
    return weights / total if total > 0 else pd.Series(dtype=float)
