from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import ConfigBundle
from ..factors.registry import build_factor_panel
from ..portfolio.construction import build_target_weights
from ..portfolio.turnover import portfolio_turnover
from ..providers.base import DataProvider
from ..research.ic import compute_ic_bundle
from ..research.quantiles import compute_quantile_forward_returns
from ..research.stability import compute_stability_bundle
from ..signals.composite_score import compute_composite_score
from ..signals.ranking import rank_and_select
from ..types import BacktestArtifacts
from ..universe.filters import build_universe
from .broker import build_price_tables, execute_rebalance, portfolio_market_value
from .calendar import build_trading_dates, generate_signal_dates, map_signal_to_execution_dates
from .cost_model import CostModel
from .performance import compute_performance_metrics


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BacktestEngine:
    config_bundle: ConfigBundle

    def run(self, provider: DataProvider) -> BacktestArtifacts:
        config = self.config_bundle
        backtest_cfg = config.backtest
        strategy_cfg = config.strategy
        market_cfg = config.market

        start = pd.Timestamp(backtest_cfg["start"])
        end = pd.Timestamp(backtest_cfg["end"])
        benchmark_symbol = str(
            backtest_cfg.get("benchmark_symbol")
            or market_cfg.get("benchmark_symbol")
            or "^HSI"
        )
        initial_capital = float(backtest_cfg.get("initial_capital", 1_000_000.0))
        frequency = str(
            backtest_cfg.get("rebalance_frequency")
            or strategy_cfg.get("signals", {}).get("rebalance_frequency")
            or "M"
        )

        cost_cfg = backtest_cfg.get("costs", {})
        cost_model = CostModel(
            transaction_cost_bps=float(cost_cfg.get("transaction_cost_bps", 20.0)),
            slippage_bps=float(cost_cfg.get("slippage_bps", 10.0)),
            minimum_fee=float(cost_cfg.get("minimum_fee", 0.0)),
        )

        price_history = provider.get_price_history(start=start, end=end)
        benchmark_history = provider.get_benchmark_history(benchmark_symbol, start=start, end=end)
        if not benchmark_history.empty:
            exists = not price_history[price_history["symbol"] == benchmark_symbol].empty
            if not exists:
                price_history = pd.concat([price_history, benchmark_history], ignore_index=True)
        if price_history.empty:
            raise ValueError("No price history available in the requested interval.")

        price_tables = build_price_tables(price_history)
        trading_dates = build_trading_dates(price_history, benchmark_symbol=benchmark_symbol)
        signal_dates = generate_signal_dates(trading_dates, frequency=frequency)
        execution_map = map_signal_to_execution_dates(trading_dates, signal_dates, lag_days=1)

        master = provider.get_security_master()
        if master.empty:
            raise ValueError("Security master is empty.")

        holdings: dict[str, int] = {}
        cash = initial_capital
        pending_targets: dict[pd.Timestamp, pd.Series] = {}
        nav_rows: list[dict[str, Any]] = []
        trade_frames: list[pd.DataFrame] = []
        holdings_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        scored_snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
        turnover_series: list[float] = []
        prev_target = pd.Series(dtype=float)

        factor_weights: dict[str, float] = strategy_cfg.get("factor_weights", {})
        if not factor_weights:
            raise ValueError("strategy.factor_weights is required.")

        transform_cfg = strategy_cfg.get("transform", {})
        factor_transform = str(transform_cfg.get("method", "zscore"))
        winsor_limits = tuple(transform_cfg.get("winsorize_limits", [0.05, 0.95]))
        min_factors_required = int(strategy_cfg.get("min_factors_required", 4))
        by_industry = bool(transform_cfg.get("by_industry", False))

        selection_cfg = strategy_cfg.get("selection", {})
        top_n = selection_cfg.get("top_n")
        top_percentile = selection_cfg.get("top_percentile")
        min_selection = int(selection_cfg.get("min_selection", 10))

        portfolio_cfg = strategy_cfg.get("portfolio", {})
        weight_method = str(portfolio_cfg.get("weight_method", "equal_weight"))
        max_weight = float(
            portfolio_cfg.get(
                "max_single_weight",
                backtest_cfg.get("max_single_weight", 1.0),
            )
        )
        min_holdings = int(portfolio_cfg.get("min_holdings", min_selection))
        max_holdings = portfolio_cfg.get("max_holdings")
        max_holdings_int = int(max_holdings) if max_holdings is not None else None

        for date in trading_dates:
            if date in signal_dates:
                trading_status = provider.get_trading_status(master["symbol"].tolist(), date)
                universe = build_universe(
                    master=master,
                    price_history=price_history[price_history["date"] <= date].copy(),
                    trading_status=trading_status,
                    as_of=date,
                    config=market_cfg.get("universe", market_cfg),
                )

                if universe.empty:
                    scored = pd.DataFrame(columns=["symbol", "composite_score"])
                    selected = scored.copy()
                    target_weights = pd.Series(dtype=float)
                else:
                    symbols = universe["symbol"].tolist()
                    financials = provider.get_financials(symbols=symbols, as_of_date=date)
                    if financials.empty:
                        scored = universe.assign(composite_score=float("nan"))
                        selected = scored.head(0)
                        target_weights = pd.Series(dtype=float)
                    else:
                        financials = financials.merge(
                            universe[["symbol", "industry"]],
                            on="symbol",
                            how="left",
                        )
                        panel = build_factor_panel(
                            financials=financials,
                            transform=factor_transform,
                            winsor_limits=(float(winsor_limits[0]), float(winsor_limits[1])),
                            by_industry=by_industry,
                        )
                        scored = compute_composite_score(
                            frame=panel,
                            weights=factor_weights,
                            min_factors_required=min_factors_required,
                        )
                        scored = universe.merge(scored, on="symbol", how="inner")
                        selected = rank_and_select(
                            scored,
                            top_n=int(top_n) if top_n is not None else None,
                            top_percentile=float(top_percentile) if top_percentile is not None else None,
                            min_selection=min_selection,
                        )
                        target_weights = build_target_weights(
                            selected=selected,
                            method=weight_method,
                            max_weight=max_weight,
                            min_holdings=min_holdings,
                            max_holdings=max_holdings_int,
                        )

                scored_snapshots[date] = scored.copy()
                turnover_series.append(portfolio_turnover(prev_target, target_weights))
                prev_target = target_weights.copy()

                for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                    selection_rows.append(
                        {
                            "signal_date": date,
                            "symbol": row["symbol"],
                            "rank": rank,
                            "target_weight": float(target_weights.get(row["symbol"], 0.0)),
                            "composite_score": float(row.get("composite_score", float("nan"))),
                            "valid_factor_count": int(row.get("valid_factor_count", 0)),
                        }
                    )

                exec_date = execution_map.get(date)
                if exec_date is not None:
                    pending_targets[exec_date] = target_weights

            if date in pending_targets:
                target_weights = pending_targets.pop(date)
                holdings, cash, trades = execute_rebalance(
                    date=date,
                    target_weights=target_weights,
                    holdings=holdings,
                    cash=cash,
                    price_tables=price_tables,
                    cost_model=cost_model,
                )
                if not trades.empty:
                    trade_frames.append(trades)

            holdings_value = portfolio_market_value(
                holdings=holdings,
                date=date,
                close_prices=price_tables["close"],
            )
            total_equity = cash + holdings_value
            benchmark_nav = _benchmark_nav(
                benchmark_symbol=benchmark_symbol,
                date=date,
                close_prices=price_tables["close"],
            )
            nav_rows.append(
                {
                    "date": date,
                    "cash": cash,
                    "holdings_value": holdings_value,
                    "total_equity": total_equity,
                    "nav": total_equity / initial_capital,
                    "benchmark_nav": benchmark_nav,
                }
            )

            for symbol, shares in holdings.items():
                if shares <= 0:
                    continue
                close_price = _last_price(symbol=symbol, date=date, close_prices=price_tables["close"])
                market_value = shares * close_price
                holdings_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "shares": shares,
                        "close_price": close_price,
                        "market_value": market_value,
                        "portfolio_weight": market_value / total_equity if total_equity else 0.0,
                    }
                )

        nav_history = pd.DataFrame(nav_rows)
        trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        holdings_history = pd.DataFrame(holdings_rows)
        selection_history = pd.DataFrame(selection_rows)
        metrics = compute_performance_metrics(nav_history=nav_history, turnover_series=turnover_series)

        factor_names = list(factor_weights.keys())
        ic_bundle = compute_ic_bundle(
            scored_snapshots=scored_snapshots,
            signal_dates=signal_dates,
            close_prices=price_tables["close"],
            factor_names=factor_names,
        )
        quantile_returns = compute_quantile_forward_returns(
            scored_snapshots=scored_snapshots,
            signal_dates=signal_dates,
            close_prices=price_tables["close"],
            score_column="composite_score",
            quantiles=int(strategy_cfg.get("quantiles", 5)),
        )
        stability_bundle = compute_stability_bundle(scored_snapshots=scored_snapshots, factor_names=factor_names)

        research_outputs = {
            "ic_timeseries": ic_bundle.get("ic_timeseries", pd.DataFrame()),
            "ic_summary": ic_bundle.get("ic_summary", pd.DataFrame()),
            "rolling_ic": ic_bundle.get("rolling_ic", pd.DataFrame()),
            "quantile_returns": quantile_returns,
            "factor_coverage": stability_bundle.get("coverage", pd.DataFrame()),
            "factor_moments": stability_bundle.get("moments", pd.DataFrame()),
            "factor_correlation": stability_bundle.get("correlation", pd.DataFrame()),
        }
        logger.info("Backtest finished with %s trading days and %s trades.", len(nav_history), len(trades))
        return BacktestArtifacts(
            nav_history=nav_history,
            trades=trades,
            holdings_history=holdings_history,
            selection_history=selection_history,
            metrics=metrics,
            research_outputs=research_outputs,
        )


def _last_price(symbol: str, date: pd.Timestamp, close_prices: pd.DataFrame) -> float:
    if symbol not in close_prices.columns:
        return 0.0
    series = close_prices[symbol].dropna().loc[:date]
    if series.empty:
        return 0.0
    return float(series.iloc[-1])


def _benchmark_nav(benchmark_symbol: str, date: pd.Timestamp, close_prices: pd.DataFrame) -> float:
    if benchmark_symbol not in close_prices.columns:
        return 1.0
    series = close_prices[benchmark_symbol].dropna()
    if series.empty:
        return 1.0
    base = float(series.iloc[0])
    if base <= 0:
        return 1.0
    current = series.loc[:date]
    if current.empty:
        return 1.0
    return float(current.iloc[-1] / base)
