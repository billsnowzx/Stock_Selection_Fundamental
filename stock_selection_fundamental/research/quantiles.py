from __future__ import annotations

import pandas as pd


def compute_quantile_forward_returns(
    scored_snapshots: dict[pd.Timestamp, pd.DataFrame],
    signal_dates: list[pd.Timestamp],
    close_prices: pd.DataFrame,
    score_column: str = "composite_score",
    quantiles: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    if quantiles < 2:
        return pd.DataFrame()

    for current_date, next_date in zip(signal_dates[:-1], signal_dates[1:]):
        snapshot = scored_snapshots.get(current_date)
        if snapshot is None or snapshot.empty:
            continue
        if "symbol" not in snapshot.columns or score_column not in snapshot.columns:
            continue

        frame = snapshot[["symbol", score_column]].copy()
        frame["forward_return"] = frame["symbol"].map(
            lambda symbol: _forward_return(symbol, current_date, next_date, close_prices)
        )
        frame = frame.dropna(subset=[score_column, "forward_return"])
        if len(frame) < quantiles:
            continue

        ranked = frame[score_column].rank(method="first")
        frame["quantile"] = pd.qcut(ranked, q=quantiles, labels=False, duplicates="drop") + 1
        grouped = frame.groupby("quantile")["forward_return"].mean()
        for quantile, value in grouped.items():
            rows.append(
                {
                    "signal_date": current_date,
                    "quantile": int(quantile),
                    "mean_forward_return": float(value),
                }
            )
    return pd.DataFrame(rows)


def _forward_return(
    symbol: str,
    current_date: pd.Timestamp,
    next_date: pd.Timestamp,
    close_prices: pd.DataFrame,
) -> float:
    if symbol not in close_prices.columns:
        return float("nan")
    start_series = close_prices[symbol].dropna().loc[:current_date]
    end_series = close_prices[symbol].dropna().loc[:next_date]
    if start_series.empty or end_series.empty:
        return float("nan")
    start_price = float(start_series.iloc[-1])
    if start_price <= 0:
        return float("nan")
    return float(end_series.iloc[-1] / start_price - 1.0)
