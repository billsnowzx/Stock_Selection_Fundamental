from __future__ import annotations

import pandas as pd


def compute_ic_bundle(
    scored_snapshots: dict[pd.Timestamp, pd.DataFrame],
    signal_dates: list[pd.Timestamp],
    close_prices: pd.DataFrame,
    factor_names: list[str],
    rolling_window: int = 12,
) -> dict[str, pd.DataFrame]:
    ic_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for current_date, next_date in zip(signal_dates[:-1], signal_dates[1:]):
        snapshot = scored_snapshots.get(current_date)
        if snapshot is None or snapshot.empty:
            continue
        if "symbol" not in snapshot.columns:
            continue

        diagnostics = snapshot.copy()
        diagnostics["forward_return"] = diagnostics["symbol"].map(
            lambda symbol: _forward_return(symbol, current_date, next_date, close_prices)
        )
        diagnostics = diagnostics.dropna(subset=["forward_return"])
        if diagnostics.empty:
            continue

        for factor in factor_names:
            if factor not in diagnostics.columns:
                continue
            values = diagnostics[factor]
            if values.notna().sum() < 3:
                continue
            ic_rows.append(
                {
                    "signal_date": current_date,
                    "factor": factor,
                    "ic": float(values.corr(diagnostics["forward_return"], method="pearson")),
                    "rank_ic": float(values.corr(diagnostics["forward_return"], method="spearman")),
                }
            )

    ic_timeseries = pd.DataFrame(ic_rows)
    if ic_timeseries.empty:
        return {
            "ic_timeseries": ic_timeseries,
            "ic_summary": pd.DataFrame(columns=["factor", "mean_ic", "mean_rank_ic"]),
            "rolling_ic": pd.DataFrame(columns=["signal_date", "factor", "rolling_ic"]),
        }

    ic_summary = (
        ic_timeseries.groupby("factor")[["ic", "rank_ic"]]
        .mean()
        .rename(columns={"ic": "mean_ic", "rank_ic": "mean_rank_ic"})
        .reset_index()
    )
    rolling = ic_timeseries.sort_values("signal_date").copy()
    rolling["rolling_ic"] = rolling.groupby("factor")["ic"].transform(
        lambda s: s.rolling(rolling_window, min_periods=1).mean()
    )
    return {
        "ic_timeseries": ic_timeseries.sort_values(["signal_date", "factor"]).reset_index(drop=True),
        "ic_summary": ic_summary.sort_values("factor").reset_index(drop=True),
        "rolling_ic": rolling[["signal_date", "factor", "rolling_ic"]].reset_index(drop=True),
    }


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
