from __future__ import annotations

import pandas as pd

from .base import FactorDefinition
from .cashflow import fcf_conversion
from .growth import net_income_growth_yoy, revenue_growth_yoy
from .leverage import debt_to_cashflow
from .profitability import net_margin, roic
from .transforms import rank_normalize, winsorize, zscore


FACTOR_DIRECTIONS = {
    "roic": "positive",
    "net_margin": "positive",
    "debt_to_cashflow": "negative",
    "revenue_growth_yoy": "positive",
    "net_income_growth_yoy": "positive",
    "fcf_conversion": "positive",
}


def get_factor_definitions() -> list[FactorDefinition]:
    return [
        FactorDefinition(name="roic", lookback="latest_available", direction="positive", compute=roic),
        FactorDefinition(name="net_margin", lookback="latest_available", direction="positive", compute=net_margin),
        FactorDefinition(name="debt_to_cashflow", lookback="latest_available", direction="negative", compute=debt_to_cashflow),
        FactorDefinition(name="revenue_growth_yoy", lookback="1y", direction="positive", compute=revenue_growth_yoy),
        FactorDefinition(name="net_income_growth_yoy", lookback="1y", direction="positive", compute=net_income_growth_yoy),
        FactorDefinition(name="fcf_conversion", lookback="latest_available", direction="positive", compute=fcf_conversion),
    ]


def _apply_transform(series: pd.Series, method: str) -> pd.Series:
    if method == "rank":
        return rank_normalize(series)
    return zscore(series)


def build_factor_panel(
    financials: pd.DataFrame,
    transform: str = "zscore",
    winsor_limits: tuple[float, float] = (0.05, 0.95),
    by_industry: bool = False,
) -> pd.DataFrame:
    output = financials.copy()

    output["roic"] = output.apply(roic, axis=1)
    output["net_margin"] = output.apply(net_margin, axis=1)
    output["debt_to_cashflow"] = output.apply(debt_to_cashflow, axis=1)
    output["revenue_growth_yoy"] = output.apply(revenue_growth_yoy, axis=1)
    output["net_income_growth_yoy"] = output.apply(net_income_growth_yoy, axis=1)
    output["fcf_conversion"] = output.apply(fcf_conversion, axis=1)

    factor_cols = list(FACTOR_DIRECTIONS)

    if by_industry and "industry" in output.columns:
        standardized = []
        for _, group in output.groupby("industry", dropna=False):
            temp = group.copy()
            for col in factor_cols:
                s = winsorize(temp[col], *winsor_limits)
                s = _apply_transform(s, transform)
                if FACTOR_DIRECTIONS[col] == "negative":
                    s = -s
                temp[f"{col}_score"] = s
            standardized.append(temp)
        output = pd.concat(standardized, ignore_index=True)
    else:
        for col in factor_cols:
            s = winsorize(output[col], *winsor_limits)
            s = _apply_transform(s, transform)
            if FACTOR_DIRECTIONS[col] == "negative":
                s = -s
            output[f"{col}_score"] = s

    return output
