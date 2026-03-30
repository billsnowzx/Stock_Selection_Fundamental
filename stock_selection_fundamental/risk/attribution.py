from __future__ import annotations

import numpy as np
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
                "interaction_component",
                "unexplained_component",
                "active_model_error",
                "total_return_recon_error",
            ]
        )

    nav = nav_history.copy().sort_values("date")
    nav["date"] = pd.to_datetime(nav["date"])
    nav["portfolio_return"] = pd.to_numeric(nav["nav"], errors="coerce").pct_change().fillna(0.0)
    nav["benchmark_return"] = pd.to_numeric(nav["benchmark_nav"], errors="coerce").pct_change().fillna(0.0)
    nav["active_return"] = nav["portfolio_return"] - nav["benchmark_return"]

    industries = security_master[["symbol", "industry_std"]].drop_duplicates("symbol")
    symbol_panel = _build_symbol_panel(
        holdings_history=holdings_history,
        price_history=price_history,
        industries=industries,
    )
    industry_daily = _industry_level_decomposition(symbol_panel)

    by_date = (
        industry_daily.groupby("date")[["industry_component", "selection_component", "interaction_component"]]
        .sum()
        .reset_index()
    )
    result = nav.merge(by_date, on="date", how="left").fillna(
        {"industry_component": 0.0, "selection_component": 0.0, "interaction_component": 0.0}
    )
    result["market_component"] = result["benchmark_return"]
    result["active_model_error"] = (
        result["active_return"]
        - result["industry_component"]
        - result["selection_component"]
        - result["interaction_component"]
    )
    result["unexplained_component"] = result["active_model_error"]
    result["total_return_recon_error"] = (
        result["portfolio_return"]
        - result["market_component"]
        - result["industry_component"]
        - result["selection_component"]
        - result["interaction_component"]
        - result["unexplained_component"]
    )

    result["cum_portfolio_return"] = (1.0 + result["portfolio_return"]).cumprod() - 1.0
    result["cum_benchmark_return"] = (1.0 + result["benchmark_return"]).cumprod() - 1.0
    result["cum_active_return"] = (1.0 + result["cum_portfolio_return"]) / (1.0 + result["cum_benchmark_return"]) - 1.0
    for col in [
        "industry_component",
        "selection_component",
        "interaction_component",
        "unexplained_component",
        "active_model_error",
    ]:
        result[f"cum_{col}"] = result[col].cumsum()

    return result.sort_values("date").reset_index(drop=True)


def summarize_attribution(attribution_daily: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "start_date",
        "end_date",
        "n_days",
        "total_portfolio_return",
        "total_benchmark_return",
        "total_active_return",
        "industry_sum",
        "selection_sum",
        "interaction_sum",
        "unexplained_sum",
        "active_model_error_abs_mean",
        "active_model_error_abs_max",
        "total_return_recon_error_abs_max",
    ]
    if attribution_daily.empty:
        return pd.DataFrame(columns=columns)

    frame = attribution_daily.copy().sort_values("date")
    start_date = pd.Timestamp(frame["date"].min())
    end_date = pd.Timestamp(frame["date"].max())
    total_portfolio = float((1.0 + frame["portfolio_return"]).prod() - 1.0)
    total_benchmark = float((1.0 + frame["benchmark_return"]).prod() - 1.0)
    total_active = float((1.0 + total_portfolio) / (1.0 + total_benchmark) - 1.0)
    summary = pd.DataFrame(
        [
            {
                "start_date": start_date,
                "end_date": end_date,
                "n_days": int(frame["date"].nunique()),
                "total_portfolio_return": total_portfolio,
                "total_benchmark_return": total_benchmark,
                "total_active_return": total_active,
                "industry_sum": float(frame["industry_component"].sum()),
                "selection_sum": float(frame["selection_component"].sum()),
                "interaction_sum": float(frame["interaction_component"].sum()),
                "unexplained_sum": float(frame["unexplained_component"].sum()),
                "active_model_error_abs_mean": float(frame["active_model_error"].abs().mean()),
                "active_model_error_abs_max": float(frame["active_model_error"].abs().max()),
                "total_return_recon_error_abs_max": float(frame["total_return_recon_error"].abs().max()),
            }
        ]
    )
    return summary


def _build_symbol_panel(
    holdings_history: pd.DataFrame,
    price_history: pd.DataFrame,
    industries: pd.DataFrame,
) -> pd.DataFrame:
    returns = _symbol_daily_returns(price_history=price_history)
    portfolio_weights = _portfolio_symbol_weights(holdings_history=holdings_history)
    benchmark_weights = _benchmark_symbol_weights(returns=returns)

    panel = (
        returns.merge(portfolio_weights, on=["date", "symbol"], how="left")
        .merge(benchmark_weights, on=["date", "symbol"], how="left")
        .merge(industries, on="symbol", how="left")
    )
    panel["industry_std"] = panel["industry_std"].fillna("UNKNOWN")
    panel["w_p"] = pd.to_numeric(panel["w_p"], errors="coerce").fillna(0.0)
    panel["w_b"] = pd.to_numeric(panel["w_b"], errors="coerce").fillna(0.0)
    panel["ret"] = pd.to_numeric(panel["ret"], errors="coerce").fillna(0.0)
    return panel


def _industry_level_decomposition(symbol_panel: pd.DataFrame) -> pd.DataFrame:
    if symbol_panel.empty:
        return pd.DataFrame(columns=["date", "industry_std", "industry_component", "selection_component", "interaction_component"])

    rows: list[dict[str, float | pd.Timestamp | str]] = []
    for (date, industry), group in symbol_panel.groupby(["date", "industry_std"]):
        row = _per_industry_row(group)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "industry_std": str(industry),
                "W_p": float(row["W_p"]),
                "W_b": float(row["W_b"]),
                "R_p": float(row["R_p"]),
                "R_b": float(row["R_b"]),
            }
        )
    grouped = pd.DataFrame(rows)
    grouped["industry_component"] = (grouped["W_p"] - grouped["W_b"]) * grouped["R_b"]
    grouped["selection_component"] = grouped["W_b"] * (grouped["R_p"] - grouped["R_b"])
    grouped["interaction_component"] = (grouped["W_p"] - grouped["W_b"]) * (grouped["R_p"] - grouped["R_b"])
    return grouped


def _per_industry_row(group: pd.DataFrame) -> pd.Series:
    wp = pd.to_numeric(group["w_p"], errors="coerce").fillna(0.0)
    wb = pd.to_numeric(group["w_b"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(group["ret"], errors="coerce").fillna(0.0)

    w_p_total = float(wp.sum())
    w_b_total = float(wb.sum())
    r_p = float(np.average(ret, weights=wp)) if w_p_total > 0 else 0.0
    r_b = float(np.average(ret, weights=wb)) if w_b_total > 0 else 0.0
    return pd.Series(
        {
            "W_p": w_p_total,
            "W_b": w_b_total,
            "R_p": r_p,
            "R_b": r_b,
        }
    )


def _symbol_daily_returns(price_history: pd.DataFrame) -> pd.DataFrame:
    if price_history.empty:
        return pd.DataFrame(columns=["date", "symbol", "ret"])
    frame = price_history[["date", "symbol", "close"]].copy()
    frame = frame[~frame["symbol"].astype(str).str.startswith("^")]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_values(["symbol", "date"])
    frame["ret"] = frame.groupby("symbol")["close"].pct_change().fillna(0.0)
    return frame[["date", "symbol", "ret"]]


def _portfolio_symbol_weights(holdings_history: pd.DataFrame) -> pd.DataFrame:
    if holdings_history.empty:
        return pd.DataFrame(columns=["date", "symbol", "w_p"])
    frame = holdings_history[["date", "symbol", "portfolio_weight"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["portfolio_weight"] = pd.to_numeric(frame["portfolio_weight"], errors="coerce").fillna(0.0)
    frame = frame.groupby(["date", "symbol"], as_index=False)["portfolio_weight"].sum()
    frame = frame.rename(columns={"portfolio_weight": "w_p"})
    total = frame.groupby("date")["w_p"].transform("sum")
    frame["w_p"] = frame["w_p"] / total.where(total > 0, 1.0)
    return frame


def _benchmark_symbol_weights(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(columns=["date", "symbol", "w_b"])
    frame = returns[["date", "symbol"]].copy()
    frame["count"] = frame.groupby("date")["symbol"].transform("count")
    frame["w_b"] = 1.0 / frame["count"].where(frame["count"] > 0, 1.0)
    return frame[["date", "symbol", "w_b"]]
