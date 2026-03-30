from __future__ import annotations

import pandas as pd

from ..signals.composite_score import compute_composite_score
from ..signals.ranking import rank_and_select
from ..types import SignalFrame


def generate_signal_dates(trading_dates: pd.DatetimeIndex, frequency: str = "M") -> list[pd.Timestamp]:
    frame = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    frame["bucket"] = frame["date"].dt.to_period(frequency)
    return list(frame.groupby("bucket")["date"].max())


def generate_rebalance_signals(
    scored_factors: pd.DataFrame,
    signal_date: pd.Timestamp,
    weights: dict[str, float],
    min_factors_required: int,
    top_n: int | None,
    top_percentile: float | None,
    min_selection: int,
) -> SignalFrame:
    candidates = compute_composite_score(scored_factors, weights, min_factors_required)
    selected = rank_and_select(candidates, top_n=top_n, top_percentile=top_percentile, min_selection=min_selection)
    return SignalFrame(signal_date=signal_date, candidates=candidates, selected=selected)
