from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


CORE_METRICS = [
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "turnover",
]


@dataclass(slots=True)
class RegressionCheckResult:
    passed: bool
    details: pd.DataFrame


def freeze_baseline_metrics(metrics: dict[str, float], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: float(metrics.get(key, 0.0)) for key in CORE_METRICS}
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def compare_baseline_metrics(
    baseline_path: str | Path,
    current_metrics: dict[str, float],
    tolerance_bps: float = 200.0,
) -> RegressionCheckResult:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    passed = True
    tolerance = float(tolerance_bps) / 10000.0
    for key in CORE_METRICS:
        b = float(baseline.get(key, 0.0))
        c = float(current_metrics.get(key, 0.0))
        diff = c - b
        ok = abs(diff) <= tolerance
        if not ok:
            passed = False
        rows.append(
            {
                "metric": key,
                "baseline": b,
                "current": c,
                "diff": diff,
                "tolerance": tolerance,
                "passed": ok,
            }
        )
    return RegressionCheckResult(passed=passed, details=pd.DataFrame(rows))
