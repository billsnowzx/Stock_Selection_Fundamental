from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..types import BacktestArtifacts
from .charts import save_drawdown_chart, save_nav_chart
from .tables import metrics_to_frame


def export_html_report(
    artifacts: BacktestArtifacts,
    output_dir: str | Path,
    config_snapshot: dict[str, Any],
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nav_chart_path = save_nav_chart(artifacts.nav_history, output_path)
    drawdown_path = save_drawdown_chart(artifacts.nav_history, output_path)
    metrics_html = metrics_to_frame(artifacts.metrics).to_html(index=False, float_format=lambda x: f"{x:.6f}")

    ic_summary = artifacts.research_outputs.get("ic_summary")
    ic_summary_html = (
        ic_summary.to_html(index=False, float_format=lambda x: f"{x:.6f}")
        if ic_summary is not None and not ic_summary.empty
        else "<p>No IC summary available.</p>"
    )
    quantile = artifacts.research_outputs.get("quantile_returns")
    quantile_html = (
        quantile.to_html(index=False, float_format=lambda x: f"{x:.6f}")
        if quantile is not None and not quantile.empty
        else "<p>No quantile return output.</p>"
    )
    monthly_selection_html = _build_monthly_selection_html(artifacts.selection_history)
    attribution = artifacts.research_outputs.get("attribution_daily")
    attribution_html = (
        attribution.tail(30).to_html(index=False, float_format=lambda x: f"{x:.6f}")
        if attribution is not None and not attribution.empty
        else "<p>No attribution output.</p>"
    )
    attribution_summary = artifacts.research_outputs.get("attribution_summary")
    attribution_summary_html = (
        attribution_summary.to_html(index=False, float_format=lambda x: f"{x:.6f}")
        if attribution_summary is not None and not attribution_summary.empty
        else "<p>No attribution summary output.</p>"
    )
    constraint_stats = artifacts.research_outputs.get("constraint_stats")
    constraint_html = (
        constraint_stats.to_html(index=False, float_format=lambda x: f"{x:.6f}")
        if constraint_stats is not None and not constraint_stats.empty
        else "<p>No constraint stats.</p>"
    )
    corporate_actions = artifacts.research_outputs.get("corporate_action_ledger")
    corporate_actions_html = (
        corporate_actions.tail(30).to_html(index=False, float_format=lambda x: f"{x:.6f}")
        if corporate_actions is not None and not corporate_actions.empty
        else "<p>No corporate action events in period.</p>"
    )

    config_text = json.dumps(config_snapshot, indent=2, ensure_ascii=False, default=str)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Backtest Report</title>
  <style>
    body {{ font-family: "Segoe UI", "PingFang SC", sans-serif; margin: 24px; color: #111; }}
    h1, h2 {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .code {{ background: #f6f8fa; border: 1px solid #ddd; padding: 10px; font-family: Consolas, monospace; font-size: 12px; white-space: pre-wrap; }}
    img {{ width: 100%; max-width: 980px; border: 1px solid #ddd; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>策略回测报告</h1>
  <h2>配置摘要</h2>
  <div class="code">{config_text}</div>
  <h2>绩效指标</h2>
  {metrics_html}
  <h2>净值曲线</h2>
  <img src="{nav_chart_path.name}" alt="nav chart" />
  <h2>回撤曲线</h2>
  <img src="{drawdown_path.name}" alt="drawdown chart" />
  <h2>IC 统计</h2>
  {ic_summary_html}
  <h2>分层收益</h2>
  {quantile_html}
  <h2>按月选股清单</h2>
  {monthly_selection_html}
  <h2>约束命中统计</h2>
  {constraint_html}
  <h2>归因摘要</h2>
  {attribution_summary_html}
  <h2>归因（近30日）</h2>
  {attribution_html}
  <h2>公司行为（近30条）</h2>
  {corporate_actions_html}
</body>
</html>"""
    report_path = output_path / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _build_monthly_selection_html(selection_history: pd.DataFrame) -> str:
    if selection_history.empty:
        return "<p>No selection history output.</p>"
    if "signal_date" not in selection_history.columns or "symbol" not in selection_history.columns:
        return "<p>No selection history output.</p>"

    frame = selection_history.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.dropna(subset=["signal_date", "symbol"])
    if frame.empty:
        return "<p>No selection history output.</p>"

    frame["rebalance_month"] = frame["signal_date"].dt.strftime("%Y-%m")
    if "rank" in frame.columns:
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame = frame.sort_values(["rebalance_month", "rank", "symbol"])
    else:
        frame = frame.sort_values(["rebalance_month", "symbol"])

    monthly = (
        frame.drop_duplicates(subset=["rebalance_month", "symbol"], keep="first")
        .groupby("rebalance_month")
        .agg(
            selected_count=("symbol", "size"),
            selected_companies=("symbol", lambda s: ", ".join(s.astype(str))),
        )
        .reset_index()
    )
    return monthly.to_html(index=False)
