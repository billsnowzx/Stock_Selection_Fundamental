from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


FactorCompute = Callable[[pd.Series, pd.Series], float]


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    name: str
    direction: str
    threshold: float
    lookback: str
    compute: FactorCompute


@dataclass(slots=True)
class SignalResult:
    as_of_date: pd.Timestamp
    scored_universe: pd.DataFrame
    selected: pd.DataFrame


@dataclass(slots=True)
class BacktestResult:
    nav_history: pd.DataFrame
    trades: pd.DataFrame
    holdings_history: pd.DataFrame
    selection_history: pd.DataFrame
    metrics: dict[str, float]
    factor_diagnostics: dict[str, pd.DataFrame] = field(default_factory=dict)

