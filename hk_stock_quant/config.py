from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class StrategyConfig:
    benchmark_symbol: str = "^HSI"
    top_n: int = 20
    rebalance_frequency: str = "M"
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 20.0
    slippage_bps: float = 10.0
    min_listing_days: int = 252
    board: str = "MAIN"
    security_type: str = "EQUITY"
    min_factors_required: int = 4
    winsorize_limits: tuple[float, float] = (0.05, 0.95)
    factor_weights: dict[str, float] = field(
        default_factory=lambda: {
            "roic": 1.0,
            "net_margin": 1.0,
            "debt_to_cashflow": 1.0,
            "revenue_growth_yoy": 1.0,
            "net_income_growth_yoy": 1.0,
            "fcf_conversion": 1.0,
        }
    )
    excluded_security_types: tuple[str, ...] = (
        "ETF",
        "REIT",
        "WARRANT",
        "CBBC",
        "DERIVATIVE",
        "PREFERENCE",
    )
    quantile_groups: int = 5

