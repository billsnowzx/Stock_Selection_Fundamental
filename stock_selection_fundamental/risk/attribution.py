from __future__ import annotations

import pandas as pd


def brinson_lite_attribution(
    nav_history: pd.DataFrame,
    holdings_history: pd.DataFrame,
    price_history: pd.DataFrame,
    security_master: pd.DataFrame,
) -> pd.DataFrame:
    if nav_history.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "portfolio_return",
                "benchmark_return",
                "active_return",
                "market_component",
                "industry_component",
                "selection_component",
                "style_component",
            ]
        )

    nav = nav_history.copy().sort_values("date")
    nav["portfolio_return"] = nav["nav"].pct_change().fillna(0.0)
    nav["benchmark_return"] = nav["benchmark_nav"].pct_change().fillna(0.0)
    nav["active_return"] = nav["portfolio_return"] - nav["benchmark_return"]

    industries = security_master[["symbol", "industry_std"]].drop_duplicates("symbol")
    industry_returns = _industry_daily_returns(price_history=price_history, industries=industries)
    industry_weights_port = _portfolio_industry_weights(holdings_history=holdings_history, industries=industries)
    industry_weights_bmk = _benchmark_industry_weights(price_history=price_history, industries=industries)

    rows: list[dict[str, float | pd.Timestamp]] = []
    for _, row in nav.iterrows():
        date = pd.Timestamp(row["date"])
        r_bmk = float(row["benchmark_return"])
        active = float(row["active_return"])
        market_component = r_bmk

        day_ind_ret = industry_returns[industry_returns["date"] == date].set_index("industry_std")["industry_return"]
        wp = industry_weights_port[industry_weights_port["date"] == date].set_index("industry_std")["weight"]
        wb = industry_weights_bmk[industry_weights_bmk["date"] == date].set_index("industry_std")["weight"]
        idx = day_ind_ret.index.union(wp.index).union(wb.index)
        day_ind_ret = day_ind_ret.reindex(idx).fillna(0.0)
        wp = wp.reindex(idx).fillna(0.0)
        wb = wb.reindex(idx).fillna(0.0)

        industry_component = float(((wp - wb) * day_ind_ret).sum())
        selection_component = float(active - industry_component)
        rows.append(
            {
                "date": date,
                "portfolio_return": float(row["portfolio_return"]),
                "benchmark_return": r_bmk,
                "active_return": active,
                "market_component": market_component,
                "industry_component": industry_component,
                "selection_component": selection_component,
                "style_component": 0.0,
            }
        )
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for col in ["market_component", "industry_component", "selection_component", "style_component", "active_return"]:
        result[f"cum_{col}"] = (1 + result[col]).cumprod() - 1
    return result


def _industry_daily_returns(price_history: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    if price_history.empty:
        return pd.DataFrame(columns=["date", "industry_std", "industry_return"])
    frame = price_history[["date", "symbol", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_values(["symbol", "date"])
    frame["ret"] = frame.groupby("symbol")["close"].pct_change()
    frame = frame.merge(industries, on="symbol", how="left")
    frame["industry_std"] = frame["industry_std"].fillna("UNKNOWN")
    grouped = frame.groupby(["date", "industry_std"])["ret"].mean().reset_index(name="industry_return")
    return grouped.dropna(subset=["industry_return"])


def _portfolio_industry_weights(holdings_history: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    if holdings_history.empty:
        return pd.DataFrame(columns=["date", "industry_std", "weight"])
    frame = holdings_history[["date", "symbol", "portfolio_weight"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.merge(industries, on="symbol", how="left")
    frame["industry_std"] = frame["industry_std"].fillna("UNKNOWN")
    return frame.groupby(["date", "industry_std"])["portfolio_weight"].sum().reset_index(name="weight")


def _benchmark_industry_weights(price_history: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    if price_history.empty:
        return pd.DataFrame(columns=["date", "industry_std", "weight"])
    frame = price_history[["date", "symbol"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[~frame["symbol"].astype(str).str.startswith("^")]
    frame = frame.merge(industries, on="symbol", how="left")
    frame["industry_std"] = frame["industry_std"].fillna("UNKNOWN")
    counts = frame.groupby(["date", "industry_std"])["symbol"].nunique().reset_index(name="count")
    total = counts.groupby("date")["count"].transform("sum")
    counts["weight"] = counts["count"] / total.where(total > 0, 1)
    return counts[["date", "industry_std", "weight"]]
