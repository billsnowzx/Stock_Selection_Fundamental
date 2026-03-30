from __future__ import annotations

import pandas as pd

from .config import StrategyConfig
from .data.provider import DataProvider
from .factors import FactorScorer, default_factor_definitions
from .types import SignalResult
from .universe import UniverseBuilder


class FundamentalTopNStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.universe_builder = UniverseBuilder(config)
        self.factor_scorer = FactorScorer(config, default_factor_definitions())

    def generate_signal(
        self,
        provider: DataProvider,
        as_of_date: str | pd.Timestamp,
    ) -> SignalResult:
        as_of = pd.Timestamp(as_of_date)
        universe = self.universe_builder.build(provider, as_of)
        if universe.empty:
            empty = pd.DataFrame(columns=["symbol", "composite_score"])
            return SignalResult(as_of_date=as_of, scored_universe=empty, selected=empty)

        symbols = universe["symbol"].tolist()
        financials = provider.get_financials(symbols, as_of)
        if financials.empty:
            empty = universe.assign(composite_score=float("nan"))
            return SignalResult(as_of_date=as_of, scored_universe=empty, selected=empty.head(0))

        trading_status = provider.get_trading_status(symbols, as_of)
        scored = self.factor_scorer.score(financials, trading_status)
        scored = universe.merge(scored, on="symbol", how="inner", suffixes=("_universe", ""))
        scored = scored.dropna(subset=["composite_score"]).sort_values(
            ["composite_score", "threshold_pass_count"],
            ascending=[False, False],
        )
        scored["rank"] = range(1, len(scored) + 1)
        selected = scored.head(self.config.top_n).copy()
        selected["target_weight"] = 1.0 / len(selected) if not selected.empty else 0.0
        return SignalResult(
            as_of_date=as_of,
            scored_universe=scored.reset_index(drop=True),
            selected=selected.reset_index(drop=True),
        )
