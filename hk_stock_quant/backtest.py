from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data.provider import DataProvider
from .strategy import FundamentalTopNStrategy
from .types import BacktestResult, SignalResult


class BacktestEngine:
    def __init__(self, config: StrategyConfig, strategy: FundamentalTopNStrategy):
        self.config = config
        self.strategy = strategy
        self._last_cash = float(config.initial_capital)

    def run(
        self,
        provider: DataProvider,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> BacktestResult:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        prices = provider.get_price_history(start=start_ts, end=end_ts)
        if prices.empty:
            raise ValueError("No price data found in the requested date range.")

        price_tables = self._build_price_tables(prices)
        trading_dates = self._resolve_trading_dates(price_tables["close"])
        if len(trading_dates) < 2:
            raise ValueError("Backtest requires at least two trading dates.")

        signal_dates = self._month_end_dates(trading_dates)
        execution_dates = {
            signal_date: trading_dates[idx + 1]
            for idx, signal_date in enumerate(trading_dates[:-1])
            if signal_date in signal_dates
        }

        holdings: dict[str, int] = {}
        cash = float(self.config.initial_capital)
        pending_signals: dict[pd.Timestamp, SignalResult] = {}
        scored_universes: dict[pd.Timestamp, pd.DataFrame] = {}
        nav_rows: list[dict[str, float | pd.Timestamp]] = []
        holding_rows: list[dict[str, float | pd.Timestamp | str]] = []
        trade_rows: list[dict[str, float | pd.Timestamp | str]] = []
        selection_rows: list[dict[str, float | pd.Timestamp | str]] = []

        for date in trading_dates:
            if date in signal_dates:
                signal = self.strategy.generate_signal(provider, date)
                scored_universes[date] = signal.scored_universe.copy()
                pending_execution = execution_dates.get(date)
                if pending_execution is not None:
                    pending_signals[pending_execution] = signal
                for _, row in signal.selected.iterrows():
                    selection_rows.append(
                        {
                            "signal_date": date,
                            "symbol": row["symbol"],
                            "rank": row["rank"],
                            "target_weight": row["target_weight"],
                            "composite_score": row["composite_score"],
                            "threshold_pass_count": row["threshold_pass_count"],
                        }
                    )

            if date in pending_signals:
                signal = pending_signals.pop(date)
                cash_box = [cash]
                trade_rows.extend(
                    self._execute_rebalance(
                        date=date,
                        signal=signal,
                        holdings=holdings,
                        cash_ref=cash_box,
                        price_tables=price_tables,
                    )
                )
                cash = float(cash_box[0])

            portfolio_value = self._portfolio_value(holdings, date, price_tables["close"])
            benchmark_nav = self._benchmark_nav(date, price_tables["close"])
            nav_rows.append(
                {
                    "date": date,
                    "cash": cash,
                    "holdings_value": portfolio_value,
                    "total_equity": cash + portfolio_value,
                    "benchmark_nav": benchmark_nav,
                }
            )

            total_equity = cash + portfolio_value
            for symbol, shares in holdings.items():
                if shares <= 0:
                    continue
                close_price = self._last_available_price(symbol, date, price_tables["close"])
                market_value = shares * close_price
                holding_rows.append(
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
        nav_history["nav"] = nav_history["total_equity"] / self.config.initial_capital
        nav_history["daily_return"] = nav_history["nav"].pct_change().fillna(0.0)
        nav_history["benchmark_return"] = nav_history["benchmark_nav"].pct_change().fillna(0.0)

        trades = pd.DataFrame(trade_rows)
        holdings_history = pd.DataFrame(holding_rows)
        selection_history = pd.DataFrame(selection_rows)
        metrics = self._compute_metrics(nav_history)
        diagnostics = self._compute_factor_diagnostics(
            scored_universes=scored_universes,
            selection_history=selection_history,
            close_prices=price_tables["close"],
            signal_dates=sorted(scored_universes),
        )
        return BacktestResult(
            nav_history=nav_history,
            trades=trades,
            holdings_history=holdings_history,
            selection_history=selection_history,
            metrics=metrics,
            factor_diagnostics=diagnostics,
        )

    def _build_price_tables(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        prices = prices.sort_values(["date", "symbol"]).copy()
        close_prices = prices.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
        open_prices = prices.pivot(index="date", columns="symbol", values="open").sort_index()
        suspended = prices.pivot(index="date", columns="symbol", values="is_suspended").sort_index()
        suspended = suspended.where(suspended.notna(), True).infer_objects(copy=False).astype(bool)
        return {"close": close_prices, "open": open_prices, "suspended": suspended}

    def _resolve_trading_dates(self, close_prices: pd.DataFrame) -> list[pd.Timestamp]:
        if self.config.benchmark_symbol in close_prices.columns:
            index = close_prices[self.config.benchmark_symbol].dropna().index
        else:
            index = close_prices.index
        return list(pd.to_datetime(index))

    def _month_end_dates(self, trading_dates: list[pd.Timestamp]) -> set[pd.Timestamp]:
        frame = pd.DataFrame({"date": trading_dates})
        frame["month"] = frame["date"].dt.to_period("M")
        return set(pd.to_datetime(frame.groupby("month")["date"].max()))

    def _execute_rebalance(
        self,
        date: pd.Timestamp,
        signal: SignalResult,
        holdings: dict[str, int],
        cash_ref: list[float],
        price_tables: dict[str, pd.DataFrame],
    ) -> list[dict[str, float | pd.Timestamp | str]]:
        cash = float(cash_ref[0])
        open_prices = price_tables["open"]
        close_prices = price_tables["close"]
        suspended = price_tables["suspended"]
        trades: list[dict[str, float | pd.Timestamp | str]] = []

        current_symbols = {symbol for symbol, shares in holdings.items() if shares > 0}
        target_symbols = set(signal.selected["symbol"].tolist())
        rebalance_symbols = current_symbols | target_symbols

        tradable_symbols = {
            symbol
            for symbol in rebalance_symbols
            if symbol in open_prices.columns
            and date in open_prices.index
            and pd.notna(open_prices.at[date, symbol])
            and not bool(suspended.at[date, symbol])
        }

        for symbol in sorted(current_symbols - target_symbols):
            if symbol not in tradable_symbols:
                continue
            shares = holdings.get(symbol, 0)
            if shares <= 0:
                continue
            sell_price = float(open_prices.at[date, symbol]) * (1 - self.config.slippage_bps / 10000)
            notional = shares * sell_price
            fee = notional * self.config.transaction_cost_bps / 10000
            cash += notional - fee
            holdings[symbol] = 0
            trades.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "SELL",
                    "shares": shares,
                    "price": sell_price,
                    "notional": notional,
                    "fee": fee,
                }
            )

        locked_value = 0.0
        for symbol in current_symbols:
            if symbol not in tradable_symbols and holdings.get(symbol, 0) > 0:
                locked_value += holdings[symbol] * self._last_available_price(symbol, date, close_prices)

        total_equity = cash + self._portfolio_value(holdings, date, close_prices)
        tradable_targets = [symbol for symbol in signal.selected["symbol"] if symbol in tradable_symbols]
        tradable_budget = max(total_equity - locked_value, 0.0)
        target_value = tradable_budget / len(tradable_targets) if tradable_targets else 0.0

        desired_shares: dict[str, int] = {}
        for symbol in tradable_targets:
            buy_price = float(open_prices.at[date, symbol]) * (1 + self.config.slippage_bps / 10000)
            desired_shares[symbol] = int(target_value // buy_price)

        for symbol in sorted(rebalance_symbols):
            if symbol not in tradable_symbols:
                continue
            current_shares = holdings.get(symbol, 0)
            target_shares = desired_shares.get(symbol, 0)
            delta = target_shares - current_shares
            if delta >= 0:
                continue
            sell_price = float(open_prices.at[date, symbol]) * (1 - self.config.slippage_bps / 10000)
            shares_to_sell = abs(delta)
            notional = shares_to_sell * sell_price
            fee = notional * self.config.transaction_cost_bps / 10000
            cash += notional - fee
            holdings[symbol] = current_shares - shares_to_sell
            trades.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "SELL",
                    "shares": shares_to_sell,
                    "price": sell_price,
                    "notional": notional,
                    "fee": fee,
                }
            )

        for symbol in sorted(rebalance_symbols):
            if symbol not in tradable_symbols:
                continue
            current_shares = holdings.get(symbol, 0)
            target_shares = desired_shares.get(symbol, 0)
            delta = target_shares - current_shares
            if delta <= 0:
                continue
            buy_price = float(open_prices.at[date, symbol]) * (1 + self.config.slippage_bps / 10000)
            unit_cost = buy_price * (1 + self.config.transaction_cost_bps / 10000)
            affordable = int(cash // unit_cost)
            shares_to_buy = min(delta, affordable)
            if shares_to_buy <= 0:
                continue
            notional = shares_to_buy * buy_price
            fee = notional * self.config.transaction_cost_bps / 10000
            cash -= notional + fee
            holdings[symbol] = current_shares + shares_to_buy
            trades.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "BUY",
                    "shares": shares_to_buy,
                    "price": buy_price,
                    "notional": notional,
                    "fee": fee,
                }
            )

        cash_ref[0] = cash
        return trades

    def _portfolio_value(self, holdings: dict[str, int], date: pd.Timestamp, close_prices: pd.DataFrame) -> float:
        value = 0.0
        for symbol, shares in holdings.items():
            if shares <= 0:
                continue
            value += shares * self._last_available_price(symbol, date, close_prices)
        return value

    def _last_available_price(self, symbol: str, date: pd.Timestamp, price_frame: pd.DataFrame) -> float:
        if symbol not in price_frame.columns:
            return 0.0
        series = price_frame[symbol].dropna().loc[:date]
        if series.empty:
            return 0.0
        return float(series.iloc[-1])

    def _benchmark_nav(self, date: pd.Timestamp, close_prices: pd.DataFrame) -> float:
        if self.config.benchmark_symbol not in close_prices.columns:
            return 1.0
        series = close_prices[self.config.benchmark_symbol].dropna().loc[:date]
        if series.empty:
            return 1.0
        base = float(close_prices[self.config.benchmark_symbol].dropna().iloc[0])
        return float(series.iloc[-1] / base)

    def _compute_metrics(self, nav_history: pd.DataFrame) -> dict[str, float]:
        nav = nav_history["nav"]
        returns = nav_history["daily_return"]
        benchmark = nav_history["benchmark_nav"]
        benchmark_returns = nav_history["benchmark_return"]
        periods = max(len(nav_history), 1)

        return_std = float(returns.std(ddof=0))
        volatility = float(return_std * np.sqrt(252))
        sharpe = 0.0 if return_std == 0 else float(returns.mean() / return_std * np.sqrt(252))
        active_returns = returns - benchmark_returns
        active_std = float(active_returns.std(ddof=0))
        info_ratio = 0.0 if active_std == 0 else float(active_returns.mean() / active_std * np.sqrt(252))
        rolling_max = nav.cummax()
        drawdown = nav / rolling_max - 1
        return {
            "total_return": float(nav.iloc[-1] - 1),
            "benchmark_return": float(benchmark.iloc[-1] - 1),
            "annualized_return": float(nav.iloc[-1] ** (252 / periods) - 1),
            "annualized_volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": float(drawdown.min()),
            "win_rate": float((returns > 0).mean()),
            "information_ratio": info_ratio,
            "excess_return": float(nav.iloc[-1] - benchmark.iloc[-1]),
        }

    def _compute_factor_diagnostics(
        self,
        scored_universes: dict[pd.Timestamp, pd.DataFrame],
        selection_history: pd.DataFrame,
        close_prices: pd.DataFrame,
        signal_dates: list[pd.Timestamp],
    ) -> dict[str, pd.DataFrame]:
        if len(signal_dates) < 2:
            return {}

        factor_names = list(self.config.factor_weights)
        ic_rows: list[dict[str, float | pd.Timestamp | str]] = []
        quantile_rows: list[dict[str, float | pd.Timestamp | int]] = []
        hit_rate_rows: list[dict[str, float | pd.Timestamp]] = []

        for current_date, next_date in zip(signal_dates[:-1], signal_dates[1:]):
            scored = scored_universes.get(current_date)
            if scored is None or scored.empty:
                continue

            required_columns = ["symbol", "composite_score", *[f"{name}_raw" for name in factor_names]]
            if not set(required_columns).issubset(scored.columns):
                continue
            diagnostics = scored[required_columns].copy()
            diagnostics["forward_return"] = diagnostics["symbol"].map(
                lambda symbol: self._forward_return(symbol, current_date, next_date, close_prices)
            )
            diagnostics = diagnostics.dropna(subset=["forward_return"])
            if diagnostics.empty:
                continue

            for factor_name in factor_names:
                series = diagnostics[f"{factor_name}_raw"]
                if series.notna().sum() < 3:
                    continue
                ic_rows.append(
                    {
                        "signal_date": current_date,
                        "factor": factor_name,
                        "ic": float(series.corr(diagnostics["forward_return"], method="pearson")),
                        "rank_ic": float(series.corr(diagnostics["forward_return"], method="spearman")),
                    }
                )

            composite = diagnostics["composite_score"]
            if composite.notna().sum() >= self.config.quantile_groups:
                diagnostics["quantile"] = pd.qcut(
                    composite.rank(method="first"),
                    q=self.config.quantile_groups,
                    labels=False,
                    duplicates="drop",
                ) + 1
                for quantile, value in diagnostics.groupby("quantile")["forward_return"].mean().items():
                    quantile_rows.append(
                        {
                            "signal_date": current_date,
                            "quantile": int(quantile),
                            "mean_forward_return": float(value),
                        }
                    )

            selected_symbols = selection_history.loc[
                selection_history["signal_date"] == current_date, "symbol"
            ]
            if not selected_symbols.empty:
                selected_returns = diagnostics[diagnostics["symbol"].isin(selected_symbols)]["forward_return"]
                if not selected_returns.empty:
                    hit_rate_rows.append(
                        {
                            "signal_date": current_date,
                            "hit_rate": float((selected_returns > 0).mean()),
                            "mean_selected_forward_return": float(selected_returns.mean()),
                        }
                    )

        ic_timeseries = pd.DataFrame(ic_rows)
        if ic_timeseries.empty:
            return {}

        ic_summary = (
            ic_timeseries.groupby("factor")[["ic", "rank_ic"]]
            .mean()
            .rename(columns={"ic": "mean_ic", "rank_ic": "mean_rank_ic"})
            .reset_index()
        )
        return {
            "ic_timeseries": ic_timeseries,
            "ic_summary": ic_summary,
            "quantile_returns": pd.DataFrame(quantile_rows),
            "hit_rate": pd.DataFrame(hit_rate_rows),
        }

    def _forward_return(
        self,
        symbol: str,
        current_date: pd.Timestamp,
        next_date: pd.Timestamp,
        close_prices: pd.DataFrame,
    ) -> float:
        if symbol not in close_prices.columns:
            return float("nan")
        start_series = close_prices[symbol].dropna().loc[:current_date]
        end_series = close_prices[symbol].dropna().loc[:next_date]
        if start_series.empty or end_series.empty or start_series.iloc[-1] <= 0:
            return float("nan")
        return float(end_series.iloc[-1] / start_series.iloc[-1] - 1)




