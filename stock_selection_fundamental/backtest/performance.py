from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_performance_metrics(
    nav_history: pd.DataFrame,
    turnover_series: list[float] | None = None,
) -> dict[str, float]:
    if nav_history.empty:
        return {}

    frame = nav_history.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["daily_return"] = frame["nav"].pct_change().fillna(0.0)
    if "benchmark_nav" in frame.columns:
        frame["benchmark_return"] = frame["benchmark_nav"].pct_change().fillna(0.0)
    else:
        frame["benchmark_nav"] = 1.0
        frame["benchmark_return"] = 0.0

    total_return = float(frame["nav"].iloc[-1] - 1.0)
    benchmark_return = float(frame["benchmark_nav"].iloc[-1] - 1.0)

    periods = max(len(frame), 1)
    annualized_return = float(frame["nav"].iloc[-1] ** (252.0 / periods) - 1.0)
    annualized_volatility = float(frame["daily_return"].std(ddof=0) * math.sqrt(252.0))
    if annualized_volatility == 0:
        sharpe = 0.0
    else:
        sharpe = float(
            frame["daily_return"].mean() / frame["daily_return"].std(ddof=0) * math.sqrt(252.0)
        )

    rolling_peak = frame["nav"].cummax()
    drawdown = frame["nav"] / rolling_peak - 1.0
    max_drawdown = float(drawdown.min())

    active = frame["daily_return"] - frame["benchmark_return"]
    tracking_error = float(active.std(ddof=0) * math.sqrt(252.0))
    information_ratio = 0.0 if tracking_error == 0 else float(active.mean() / active.std(ddof=0) * math.sqrt(252.0))

    monthly = frame.set_index("date")[["daily_return", "benchmark_return"]].resample("ME").sum()
    monthly_win_rate = float((monthly["daily_return"] > monthly["benchmark_return"]).mean()) if not monthly.empty else 0.0

    turnover = float(np.mean(turnover_series)) if turnover_series else 0.0
    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": float(total_return - benchmark_return),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "information_ratio": information_ratio,
        "turnover": turnover,
        "monthly_win_rate": monthly_win_rate,
    }
