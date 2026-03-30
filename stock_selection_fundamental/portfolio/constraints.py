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


def enforce_non_tradable_filter(weights: pd.Series, tradable_symbols: set[str]) -> pd.Series:
    output = weights[weights.index.isin(tradable_symbols)].copy()
    total = output.sum()
    return output / total if total > 0 else output


def enforce_liquidity_constraint(
    weights: pd.Series,
    avg_turnover: pd.Series | None,
    portfolio_value: float,
    max_adv_participation: float,
) -> pd.Series:
    if weights.empty or avg_turnover is None or avg_turnover.empty:
        return weights
    output = weights.copy()
    adv = avg_turnover.reindex(output.index).fillna(0.0)
    max_notional = adv * float(max_adv_participation)
    current_notional = output * float(portfolio_value)
    clipped_notional = current_notional.where(current_notional <= max_notional, max_notional)
    clipped_weights = clipped_notional / max(float(portfolio_value), 1.0)
    total = clipped_weights.sum()
    return clipped_weights / total if total > 0 else pd.Series(dtype=float)


def enforce_style_soft_constraints(
    weights: pd.Series,
    style_exposure: pd.DataFrame | None,
    style_limits: dict[str, float] | None,
) -> pd.Series:
    """Soft constraint placeholder: trims weights on names breaching single-name style caps."""
    if weights.empty or style_exposure is None or style_exposure.empty or not style_limits:
        return weights
    output = weights.copy()
    indexed = style_exposure.set_index("symbol")
    for style, limit in style_limits.items():
        if style not in indexed.columns:
            continue
        breached = indexed[indexed[style].abs() > float(limit)].index.intersection(output.index)
        if len(breached) > 0:
            output.loc[breached] = output.loc[breached] * 0.8
    total = output.sum()
    return output / total if total > 0 else pd.Series(dtype=float)


def enforce_turnover_cap(
    previous: pd.Series,
    target: pd.Series,
    max_turnover: float | None,
) -> pd.Series:
    """Blend target towards previous allocation to satisfy turnover cap."""
    if target.empty:
        return target
    if max_turnover is None:
        return target
    cap = float(max_turnover)
    if cap < 0:
        cap = 0.0

    all_symbols = previous.index.union(target.index)
    prev_aligned = previous.reindex(all_symbols).fillna(0.0)
    tgt_aligned = target.reindex(all_symbols).fillna(0.0)
    turnover = float((tgt_aligned - prev_aligned).abs().sum() / 2.0)
    if turnover <= cap or turnover <= 0:
        total = tgt_aligned.sum()
        return tgt_aligned / total if total > 0 else pd.Series(dtype=float)

    alpha = cap / turnover
    blended = prev_aligned + alpha * (tgt_aligned - prev_aligned)
    blended = blended.clip(lower=0.0)
    total = blended.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return blended / total
