from __future__ import annotations

import pandas as pd


def neutralize_by_industry(
    weights: pd.Series,
    symbol_industry: pd.Series,
    benchmark_industry_weights: pd.Series | None = None,
) -> pd.Series:
    if weights.empty:
        return weights
    industries = symbol_industry.reindex(weights.index).fillna("UNKNOWN")
    grouped = weights.groupby(industries).sum()
    if benchmark_industry_weights is None or benchmark_industry_weights.empty:
        target_group = pd.Series(1.0 / len(grouped), index=grouped.index)
    else:
        target_group = benchmark_industry_weights.reindex(grouped.index).fillna(0.0)
        if target_group.sum() <= 0:
            target_group = pd.Series(1.0 / len(grouped), index=grouped.index)
        else:
            target_group = target_group / target_group.sum()

    adjusted = weights.copy()
    for industry, current_sum in grouped.items():
        members = industries[industries == industry].index
        if current_sum <= 0:
            if len(members) == 0:
                continue
            adjusted.loc[members] = target_group[industry] / len(members)
            continue
        scale = target_group[industry] / current_sum
        adjusted.loc[members] = adjusted.loc[members] * scale
    total = adjusted.sum()
    return adjusted / total if total > 0 else adjusted


def neutralize_placeholder(weights: pd.Series) -> pd.Series:
    if weights.empty:
        return weights
    total = weights.sum()
    return weights / total if total > 0 else weights
