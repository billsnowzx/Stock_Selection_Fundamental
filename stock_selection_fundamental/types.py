from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class SignalFrame:
    signal_date: pd.Timestamp
    candidates: pd.DataFrame
    selected: pd.DataFrame


@dataclass(slots=True)
class PortfolioTarget:
    date: pd.Timestamp
    weights: pd.Series


@dataclass(slots=True)
class BacktestArtifacts:
    nav_history: pd.DataFrame
    trades: pd.DataFrame
    holdings_history: pd.DataFrame
    selection_history: pd.DataFrame
    metrics: dict[str, float]
    research_outputs: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyConfig:
    universe: dict
    rebalance_frequency: str
    selection: dict
    portfolio: dict
    costs: dict
    benchmark_symbol: str
    factor_weights: dict[str, float]
