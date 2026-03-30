from __future__ import annotations

import pandas as pd


def build_trading_dates(price_history: pd.DataFrame, benchmark_symbol: str | None = None) -> list[pd.Timestamp]:
    if price_history.empty:
        return []
    frame = price_history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if benchmark_symbol:
        benchmark = frame[frame["symbol"] == benchmark_symbol]["date"].drop_duplicates().sort_values()
        if not benchmark.empty:
            return list(benchmark)
    return list(frame["date"].drop_duplicates().sort_values())


def generate_signal_dates(trading_dates: list[pd.Timestamp], frequency: str = "M") -> list[pd.Timestamp]:
    if not trading_dates:
        return []
    frame = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    frame["bucket"] = frame["date"].dt.to_period(frequency)
    return list(frame.groupby("bucket")["date"].max())


def map_signal_to_execution_dates(
    trading_dates: list[pd.Timestamp],
    signal_dates: list[pd.Timestamp],
    lag_days: int = 1,
) -> dict[pd.Timestamp, pd.Timestamp]:
    if lag_days <= 0:
        raise ValueError("lag_days must be positive")
    execution_map: dict[pd.Timestamp, pd.Timestamp] = {}
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
    for signal_date in signal_dates:
        idx = date_to_idx.get(signal_date)
        if idx is None:
            continue
        exec_idx = idx + lag_days
        if exec_idx < len(trading_dates):
            execution_map[signal_date] = trading_dates[exec_idx]
    return execution_map
