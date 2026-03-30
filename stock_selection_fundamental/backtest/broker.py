from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .cost_model import CostModel


def build_price_tables(price_history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    history = price_history.sort_values(["date", "symbol"]).copy()
    close_prices = history.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
    open_prices = history.pivot(index="date", columns="symbol", values="open").sort_index()
    suspended = history.pivot(index="date", columns="symbol", values="is_suspended").sort_index()
    suspended = suspended.where(suspended.notna(), True).infer_objects(copy=False).astype(bool)
    return {"open": open_prices, "close": close_prices, "suspended": suspended}


def last_available_price(symbol: str, date: pd.Timestamp, close_prices: pd.DataFrame) -> float:
    if symbol not in close_prices.columns:
        return 0.0
    series = close_prices[symbol].dropna().loc[:date]
    if series.empty:
        return 0.0
    return float(series.iloc[-1])


def portfolio_market_value(
    holdings: Mapping[str, int],
    date: pd.Timestamp,
    close_prices: pd.DataFrame,
) -> float:
    value = 0.0
    for symbol, shares in holdings.items():
        if shares <= 0:
            continue
        value += shares * last_available_price(symbol, date, close_prices)
    return float(value)


def execute_rebalance(
    date: pd.Timestamp,
    target_weights: pd.Series,
    holdings: dict[str, int],
    cash: float,
    price_tables: dict[str, pd.DataFrame],
    cost_model: CostModel,
    lot_sizes: pd.Series | None = None,
    min_trade_notional: float = 0.0,
    return_details: bool = False,
) -> tuple[dict[str, int], float, pd.DataFrame] | tuple[dict[str, int], float, pd.DataFrame, dict[str, object]]:
    open_prices = price_tables["open"]
    close_prices = price_tables["close"]
    suspended = price_tables["suspended"]

    target_weights = target_weights[target_weights > 0].copy()
    if target_weights.sum() > 0:
        target_weights = target_weights / target_weights.sum()
    lot_sizes = lot_sizes if lot_sizes is not None else pd.Series(dtype=int)

    all_symbols = set(holdings) | set(target_weights.index)
    current_symbols = {symbol for symbol, shares in holdings.items() if shares > 0}
    target_symbols = set(target_weights.index)
    tradable_symbols = {
        symbol
        for symbol in all_symbols
        if symbol in open_prices.columns
        and date in open_prices.index
        and pd.notna(open_prices.at[date, symbol])
        and not bool(suspended.at[date, symbol])
    }
    non_tradable_target_symbols = sorted(target_symbols - tradable_symbols)

    trades: list[dict[str, float | str | pd.Timestamp | int]] = []

    for symbol in sorted(current_symbols - target_symbols):
        if symbol not in tradable_symbols:
            continue
        shares = int(holdings.get(symbol, 0))
        if shares <= 0:
            continue
        sell_price = cost_model.apply_sell_slippage(float(open_prices.at[date, symbol]))
        notional = shares * sell_price
        if notional < min_trade_notional:
            continue
        fee = cost_model.fee(notional)
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
                "reason": "liquidate_removed",
            }
        )

    total_equity = float(cash) + portfolio_market_value(holdings=holdings, date=date, close_prices=close_prices)
    locked_value = 0.0
    for symbol in current_symbols:
        if symbol not in tradable_symbols and holdings.get(symbol, 0) > 0:
            locked_value += holdings[symbol] * last_available_price(symbol, date, close_prices)

    tradable_target_weights = target_weights[target_weights.index.isin(tradable_symbols)]
    if tradable_target_weights.sum() > 0:
        tradable_target_weights = tradable_target_weights / tradable_target_weights.sum()
    tradable_budget = max(total_equity - locked_value, 0.0)

    desired_shares: dict[str, int] = {}
    for symbol, target_weight in tradable_target_weights.items():
        raw_open = float(open_prices.at[date, symbol])
        buy_price = cost_model.apply_buy_slippage(raw_open)
        desired = int((tradable_budget * float(target_weight)) // buy_price)
        desired_shares[symbol] = _round_down_lot(desired, int(lot_sizes.get(symbol, 1)))

    for symbol in sorted(tradable_symbols):
        current_shares = int(holdings.get(symbol, 0))
        target_shares = int(desired_shares.get(symbol, 0))
        delta = target_shares - current_shares
        if delta >= 0:
            continue
        shares_to_sell = abs(delta)
        lot_size = int(lot_sizes.get(symbol, 1))
        shares_to_sell = _round_down_lot(shares_to_sell, lot_size)
        if shares_to_sell <= 0:
            continue
        sell_price = cost_model.apply_sell_slippage(float(open_prices.at[date, symbol]))
        notional = shares_to_sell * sell_price
        if notional < min_trade_notional:
            continue
        fee = cost_model.fee(notional)
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
                "reason": "rebalance_down",
            }
        )

    for symbol in sorted(tradable_symbols):
        current_shares = int(holdings.get(symbol, 0))
        target_shares = int(desired_shares.get(symbol, 0))
        delta = target_shares - current_shares
        if delta <= 0:
            continue
        lot_size = int(lot_sizes.get(symbol, 1))
        buy_price = cost_model.apply_buy_slippage(float(open_prices.at[date, symbol]))
        per_share_cash = buy_price * (1 + cost_model.transaction_cost_bps / 10000.0)
        affordable = int(cash // per_share_cash)
        shares_to_buy = _round_down_lot(min(delta, max(affordable, 0)), lot_size)
        if shares_to_buy <= 0:
            continue
        notional = shares_to_buy * buy_price
        if notional < min_trade_notional:
            continue
        fee = cost_model.fee(notional)
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
                "reason": "rebalance_up",
            }
        )

    trade_frame = pd.DataFrame(trades)
    if not trade_frame.empty:
        trade_frame = trade_frame.sort_values(["date", "symbol", "side"]).reset_index(drop=True)
    if not return_details:
        return holdings, float(cash), trade_frame
    details = {
        "non_tradable_target_symbols": non_tradable_target_symbols,
        "unfilled_target_weight": float(target_weights.reindex(non_tradable_target_symbols).fillna(0.0).sum()),
        "executed_trade_count": int(len(trade_frame)),
    }
    return holdings, float(cash), trade_frame, details


def _round_down_lot(shares: int, lot_size: int) -> int:
    if shares <= 0:
        return 0
    lot = max(int(lot_size), 1)
    return (int(shares) // lot) * lot
