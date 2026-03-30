from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CostModel:
    transaction_cost_bps: float = 20.0
    slippage_bps: float = 10.0
    minimum_fee: float = 0.0

    def apply_buy_slippage(self, price: float) -> float:
        return float(price) * (1.0 + self.slippage_bps / 10000.0)

    def apply_sell_slippage(self, price: float) -> float:
        return float(price) * (1.0 - self.slippage_bps / 10000.0)

    def fee(self, notional: float) -> float:
        variable_fee = float(notional) * self.transaction_cost_bps / 10000.0
        return max(variable_fee, self.minimum_fee)
