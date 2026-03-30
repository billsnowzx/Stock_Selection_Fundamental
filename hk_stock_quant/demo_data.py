from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_demo_dataset(
    output_dir: str | Path,
    start: str = "2021-01-01",
    end: str = "2025-12-31",
    n_symbols: int = 30,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trading_dates = pd.bdate_range(start=start, end=end)
    report_dates = pd.date_range("2020-03-31", end=end, freq="QE-DEC")
    symbols = [f"{idx:04d}.HK" for idx in range(1, n_symbols + 1)]
    industries = [
        "Technology",
        "Consumer",
        "Healthcare",
        "Industrials",
        "Materials",
        "Utilities",
    ]

    security_rows: list[dict[str, object]] = []
    price_frames: list[pd.DataFrame] = []
    financial_rows: list[dict[str, object]] = []
    release_rows: list[dict[str, object]] = []

    quality_scores = np.linspace(0.25, 0.95, n_symbols)
    quality_scores = np.clip(quality_scores + rng.normal(0, 0.03, n_symbols), 0.05, 0.99)

    for idx, symbol in enumerate(symbols, start=1):
        quality = float(quality_scores[idx - 1])
        industry = industries[(idx - 1) % len(industries)]
        list_date = pd.Timestamp("2021-01-04")
        if idx > n_symbols - 2:
            list_date = pd.Timestamp("2024-08-01")

        security_rows.append(
            {
                "symbol": symbol,
                "name": f"Demo Holdings {idx}",
                "board": "MAIN",
                "security_type": "EQUITY",
                "industry": industry,
                "list_date": list_date,
                "delist_date": pd.NaT,
            }
        )

        active_dates = trading_dates[trading_dates >= list_date]
        base_price = 4.0 + idx * 0.8
        drift = 0.00012 + quality * 0.00045
        volatility = 0.018 - quality * 0.006
        noise = rng.normal(drift, volatility, len(active_dates))
        close = base_price * np.cumprod(1 + noise)
        overnight = rng.normal(0, 0.004, len(active_dates))
        open_price = np.insert(close[:-1], 0, close[0]) * (1 + overnight)
        high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0.003, 0.002, len(active_dates))))
        low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0.003, 0.002, len(active_dates))))
        volume = rng.integers(200_000, 2_500_000, len(active_dates))
        turnover = close * volume
        suspended = np.zeros(len(active_dates), dtype=bool)

        if len(active_dates) > 120:
            suspension_count = max(1, int(len(active_dates) * (0.001 + (1 - quality) * 0.002)))
            suspension_idx = rng.choice(len(active_dates), size=suspension_count, replace=False)
            suspended[suspension_idx] = True
            open_price[suspended] = np.nan
            high[suspended] = np.nan
            low[suspended] = np.nan
            volume[suspended] = 0
            turnover[suspended] = 0.0

        price_frames.append(
            pd.DataFrame(
                {
                    "date": active_dates,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "turnover": turnover,
                    "is_suspended": suspended,
                }
            )
        )

        base_revenue = 400 + idx * 28
        margin = 0.04 + quality * 0.18
        cashflow_ratio = 0.7 + quality * 0.55
        capex_ratio = 0.03 + (1 - quality) * 0.04
        growth = -0.01 + quality * 0.18
        roic = 0.06 + quality * 0.18
        debt_ratio = 4.6 - quality * 3.4
        tax_rate = 0.16

        for quarter_idx, period_end in enumerate(report_dates):
            years_elapsed = quarter_idx / 4
            seasonality = 1 + 0.03 * np.sin(quarter_idx / 2)
            revenue = base_revenue * ((1 + growth) ** years_elapsed) * seasonality
            revenue *= 1 + rng.normal(0, 0.015)
            net_income = revenue * margin * (1 + rng.normal(0, 0.02))
            operating_cashflow = net_income * cashflow_ratio * (1 + rng.normal(0, 0.02))
            capital_expenditure = revenue * capex_ratio
            free_cashflow = operating_cashflow - capital_expenditure
            nopat = revenue * min(margin + 0.015, 0.32)
            invested_capital = nopat / roic
            liabilities = operating_cashflow * debt_ratio
            ebit = nopat / (1 - tax_rate)
            release_date = period_end + pd.Timedelta(days=45 + int(rng.integers(0, 10)))

            financial_rows.append(
                {
                    "symbol": symbol,
                    "period_end": period_end,
                    "report_type": "QUARTERLY",
                    "revenue": revenue,
                    "net_income": net_income,
                    "ebit": ebit,
                    "effective_tax_rate": tax_rate,
                    "invested_capital": invested_capital,
                    "total_liabilities": liabilities,
                    "operating_cashflow": operating_cashflow,
                    "capital_expenditure": capital_expenditure,
                    "free_cashflow": free_cashflow,
                    "nopat": nopat,
                }
            )
            release_rows.append(
                {
                    "symbol": symbol,
                    "period_end": period_end,
                    "release_date": release_date,
                }
            )

    benchmark_noise = rng.normal(0.00018, 0.011, len(trading_dates))
    benchmark_close = 20_000 * np.cumprod(1 + benchmark_noise)
    benchmark_open = np.insert(benchmark_close[:-1], 0, benchmark_close[0])
    benchmark_frame = pd.DataFrame(
        {
            "date": trading_dates,
            "symbol": "^HSI",
            "open": benchmark_open,
            "high": np.maximum(benchmark_open, benchmark_close) * 1.002,
            "low": np.minimum(benchmark_open, benchmark_close) * 0.998,
            "close": benchmark_close,
            "volume": 0,
            "turnover": 0.0,
            "is_suspended": False,
        }
    )

    security_master = pd.DataFrame(security_rows)
    price_history = pd.concat(price_frames + [benchmark_frame], ignore_index=True)
    financials = pd.DataFrame(financial_rows)
    release_calendar = pd.DataFrame(release_rows)

    security_master.to_csv(output_path / "security_master.csv", index=False)
    price_history.to_csv(output_path / "price_history.csv", index=False)
    financials.to_csv(output_path / "financials.csv", index=False)
    release_calendar.to_csv(output_path / "release_calendar.csv", index=False)

