from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .types import FactorDefinition


def _safe_positive_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator):
        return float("nan")
    if numerator <= 0 or denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _safe_growth(current: float, previous: float) -> float:
    if pd.isna(current) or pd.isna(previous):
        return float("nan")
    if current <= 0 or previous <= 0:
        return float("nan")
    return (float(current) - float(previous)) / float(previous)


def _pick_numeric(series: pd.Series, *candidates: str) -> float:
    for candidate in candidates:
        value = series.get(candidate)
        if value is None or pd.isna(value):
            continue
        return float(value)
    return float("nan")


def default_factor_definitions() -> list[FactorDefinition]:
    return [
        FactorDefinition(
            name="roic",
            direction="positive",
            threshold=0.15,
            lookback="latest_available",
            compute=lambda f, _: _pick_numeric(f, "roic")
            if pd.notna(_pick_numeric(f, "roic"))
            else _safe_positive_ratio(f.get("nopat"), f.get("invested_capital")),
        ),
        FactorDefinition(
            name="net_margin",
            direction="positive",
            threshold=0.10,
            lookback="latest_available",
            compute=lambda f, _: _pick_numeric(f, "net_margin")
            if pd.notna(_pick_numeric(f, "net_margin"))
            else _safe_positive_ratio(f.get("net_income"), f.get("revenue")),
        ),
        FactorDefinition(
            name="debt_to_cashflow",
            direction="negative",
            threshold=3.0,
            lookback="latest_available",
            compute=lambda f, _: _pick_numeric(f, "debt_to_cashflow")
            if pd.notna(_pick_numeric(f, "debt_to_cashflow"))
            else _safe_positive_ratio(f.get("total_liabilities"), f.get("operating_cashflow")),
        ),
        FactorDefinition(
            name="revenue_growth_yoy",
            direction="positive",
            threshold=0.07,
            lookback="1y",
            compute=lambda f, _: _pick_numeric(f, "revenue_growth_yoy")
            if pd.notna(_pick_numeric(f, "revenue_growth_yoy"))
            else _safe_growth(f.get("revenue"), f.get("prev_revenue")),
        ),
        FactorDefinition(
            name="net_income_growth_yoy",
            direction="positive",
            threshold=0.09,
            lookback="1y",
            compute=lambda f, _: _pick_numeric(f, "net_income_growth_yoy")
            if pd.notna(_pick_numeric(f, "net_income_growth_yoy"))
            else _safe_growth(f.get("net_income"), f.get("prev_net_income")),
        ),
        FactorDefinition(
            name="fcf_conversion",
            direction="positive",
            threshold=0.90,
            lookback="latest_available",
            compute=lambda f, _: _pick_numeric(f, "fcf_conversion")
            if pd.notna(_pick_numeric(f, "fcf_conversion"))
            else _safe_positive_ratio(f.get("free_cashflow"), f.get("net_income")),
        ),
    ]


@dataclass(slots=True)
class FactorScorer:
    config: StrategyConfig
    factors: list[FactorDefinition]

    def score(self, financials: pd.DataFrame, market_snapshot: pd.DataFrame) -> pd.DataFrame:
        if financials.empty:
            return financials.copy()

        merged = financials.merge(
            market_snapshot.drop(columns=["date"], errors="ignore"),
            on="symbol",
            how="left",
            suffixes=("", "_market"),
        )
        raw = self._compute_raw_factors(merged)
        standardized = self._standardize(raw)

        weights = pd.Series(self.config.factor_weights, dtype=float)
        available_weights = standardized[list(weights.index)].notna().mul(weights, axis=1)
        weighted_scores = standardized[list(weights.index)].mul(weights, axis=1)
        denominator = available_weights.sum(axis=1)
        composite = weighted_scores.sum(axis=1) / denominator.replace(0, np.nan)
        valid_factor_count = standardized[list(weights.index)].notna().sum(axis=1)
        pass_count = self._threshold_pass_count(raw)

        scored = pd.concat(
            [
                merged.reset_index(drop=True),
                raw.add_suffix("_raw"),
                standardized.add_suffix("_score"),
            ],
            axis=1,
        )
        scored["valid_factor_count"] = valid_factor_count
        scored["threshold_pass_count"] = pass_count
        scored["composite_score"] = composite.where(
            valid_factor_count >= self.config.min_factors_required
        )
        return scored.sort_values("composite_score", ascending=False).reset_index(drop=True)

    def _compute_raw_factors(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for _, row in frame.iterrows():
            result = {"symbol": row["symbol"]}
            market_row = row
            for factor in self.factors:
                result[factor.name] = factor.compute(row, market_row)
            rows.append(result)
        return pd.DataFrame(rows).drop(columns=["symbol"])

    def _standardize(self, raw_factors: pd.DataFrame) -> pd.DataFrame:
        standardized = pd.DataFrame(index=raw_factors.index)
        lower, upper = self.config.winsorize_limits
        for factor in self.factors:
            series = raw_factors[factor.name].astype(float)
            valid = series.dropna()
            if valid.empty:
                standardized[factor.name] = np.nan
                continue
            clipped = series.clip(valid.quantile(lower), valid.quantile(upper))
            mean = clipped.mean()
            std = clipped.std(ddof=0)
            if pd.isna(std) or std == 0:
                zscore = pd.Series(0.0, index=series.index)
            else:
                zscore = (clipped - mean) / std
            if factor.direction == "negative":
                zscore = -zscore
            standardized[factor.name] = zscore.where(series.notna())
        return standardized

    def _threshold_pass_count(self, raw_factors: pd.DataFrame) -> pd.Series:
        counts = pd.Series(0, index=raw_factors.index, dtype=int)
        for factor in self.factors:
            values = raw_factors[factor.name]
            if factor.direction == "negative":
                passed = values < factor.threshold
            else:
                passed = values > factor.threshold
            counts = counts + passed.fillna(False).astype(int)
        return counts
