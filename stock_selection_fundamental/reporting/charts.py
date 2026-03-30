from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_nav_chart(nav_history: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / "nav_vs_benchmark.png"

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(nav_history["date"], nav_history["nav"], label="Strategy NAV")
    if "benchmark_nav" in nav_history.columns:
        axis.plot(nav_history["date"], nav_history["benchmark_nav"], label="Benchmark NAV", linestyle="--")
    axis.set_title("Strategy vs Benchmark NAV")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return chart_path


def save_drawdown_chart(nav_history: pd.DataFrame, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = output_path / "drawdown.png"

    frame = nav_history.copy()
    frame["drawdown"] = frame["nav"] / frame["nav"].cummax() - 1.0
    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.plot(frame["date"], frame["drawdown"], color="#c0392b")
    axis.fill_between(frame["date"], frame["drawdown"], 0.0, color="#e74c3c", alpha=0.2)
    axis.set_title("Drawdown")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return chart_path
