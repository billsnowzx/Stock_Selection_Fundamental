from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import matplotlib.pyplot as plt

from .config import StrategyConfig
from .types import BacktestResult


def export_backtest_report(
    result: BacktestResult,
    output_dir: str | Path,
    config: StrategyConfig,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result.nav_history.to_csv(output_path / "nav_history.csv", index=False)
    result.trades.to_csv(output_path / "trades.csv", index=False)
    result.holdings_history.to_csv(output_path / "holdings_history.csv", index=False)
    result.selection_history.to_csv(output_path / "selection_history.csv", index=False)

    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2, ensure_ascii=False)
    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)

    for name, frame in result.factor_diagnostics.items():
        frame.to_csv(output_path / f"{name}.csv", index=False)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(result.nav_history["date"], result.nav_history["nav"], label="Strategy NAV")
    axis.plot(
        result.nav_history["date"],
        result.nav_history["benchmark_nav"],
        label="Benchmark NAV",
        linestyle="--",
    )
    axis.set_title("Strategy vs Benchmark")
    axis.set_ylabel("NAV")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path / "nav_vs_benchmark.png", dpi=150)
    plt.close(figure)
    return output_path
