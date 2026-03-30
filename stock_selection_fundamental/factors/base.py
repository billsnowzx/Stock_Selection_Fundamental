from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    name: str
    lookback: str
    direction: str
    compute: Callable[[pd.Series], float]


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator):
        return float("nan")
    if denominator <= 0 or numerator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)
