from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
</body>
</html>"""
    report_path = output_path / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
