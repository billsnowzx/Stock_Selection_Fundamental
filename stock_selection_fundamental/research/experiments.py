from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json

import pandas as pd

from ..backtest.engine import BacktestEngine
from ..config import ConfigBundle, deep_merge, load_config_bundle, load_yaml
from ..providers.local_csv import LocalCSVDataProvider
from ..reporting.export_csv import export_csv_outputs
from ..reporting.export_html import export_html_report
from ..runtime import generate_run_id, hash_config_bundle, hash_data_dir


@dataclass(slots=True)
class ExperimentRunResult:
    experiment_id: str
    output_dir: Path
    summary: pd.DataFrame


def run_experiment_suite(config_path: str | Path) -> ExperimentRunResult:
    cfg_path = Path(config_path).resolve()
    cfg = load_yaml(cfg_path)
    base_backtest_cfg = cfg.get("base_backtest_config", "configs/backtests/hk_top20.yaml")
    base_bundle = load_config_bundle(base_backtest_cfg)
    output_root = Path(cfg.get("output_root", "outputs/experiments"))
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_id = str(cfg.get("experiment_id") or generate_run_id(prefix="exp"))
    exp_dir = output_root / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _expand_scenarios(cfg.get("grid", {}))
    scenario_rows: list[dict[str, object]] = []

    for idx, scenario in enumerate(scenarios, start=1):
        scenario_name = scenario.get("name") or f"scenario_{idx:03d}"
        bundle = _apply_scenario_overrides(base_bundle, scenario)
        provider = LocalCSVDataProvider(bundle.backtest.get("data_dir", "sample_data"))
        engine = BacktestEngine(bundle)
        artifacts = engine.run(provider)

        scenario_dir = exp_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        run_metadata = {
            "experiment_id": experiment_id,
            "scenario_name": scenario_name,
            "config_hash": hash_config_bundle(bundle),
            "data_hash": hash_data_dir(bundle.backtest.get("data_dir", "sample_data")),
        }
        export_csv_outputs(
            artifacts=artifacts,
            output_dir=scenario_dir,
            config_snapshot=bundle.as_dict(),
            run_metadata=run_metadata,
            write_parquet=bool(bundle.backtest.get("storage", {}).get("write_parquet", False)),
        )
        export_html_report(artifacts=artifacts, output_dir=scenario_dir, config_snapshot=bundle.as_dict())

        row = {
            "scenario_name": scenario_name,
            "total_return": artifacts.metrics.get("total_return", 0.0),
            "annualized_return": artifacts.metrics.get("annualized_return", 0.0),
            "sharpe": artifacts.metrics.get("sharpe", 0.0),
            "max_drawdown": artifacts.metrics.get("max_drawdown", 0.0),
            "turnover": artifacts.metrics.get("turnover", 0.0),
            "overrides": json.dumps(scenario, ensure_ascii=False, default=str),
        }
        scenario_rows.append(row)

        _run_walk_forward(exp_dir=exp_dir, base_bundle=bundle, scenario_name=scenario_name, cfg=cfg)
        _run_regimes(exp_dir=exp_dir, base_bundle=bundle, scenario_name=scenario_name, cfg=cfg)

    summary = pd.DataFrame(scenario_rows).sort_values(["sharpe", "annualized_return"], ascending=False)
    summary_path = exp_dir / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)
    (exp_dir / "experiment_config_snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return ExperimentRunResult(experiment_id=experiment_id, output_dir=exp_dir, summary=summary)


def _expand_scenarios(grid: dict[str, list[object]]) -> list[dict[str, object]]:
    if not grid:
        return [{"name": "default"}]
    keys = sorted(grid.keys())
    values = [grid[key] if isinstance(grid[key], list) else [grid[key]] for key in keys]
    scenarios: list[dict[str, object]] = []
    for combo in product(*values):
        scenario = {key: value for key, value in zip(keys, combo)}
        scenario["name"] = "__".join(f"{key}-{value}" for key, value in scenario.items() if key != "name")
        scenarios.append(scenario)
    return scenarios


def _apply_scenario_overrides(base: ConfigBundle, scenario: dict[str, object]) -> ConfigBundle:
    market = dict(base.market)
    strategy = dict(base.strategy)
    backtest = dict(base.backtest)
    risk = dict(base.risk)

    for key, value in scenario.items():
        if key == "name":
            continue
        if key.startswith("factor_weights."):
            factor = key.split(".", 1)[1]
            strategy.setdefault("factor_weights", {})
            strategy["factor_weights"][factor] = float(value)
        elif key.startswith("selection."):
            field = key.split(".", 1)[1]
            strategy.setdefault("selection", {})
            strategy["selection"][field] = value
        elif key.startswith("transform."):
            field = key.split(".", 1)[1]
            strategy.setdefault("transform", {})
            strategy["transform"][field] = value
        elif key.startswith("risk."):
            field = key.split(".", 1)[1]
            risk[field] = value
        elif key.startswith("backtest."):
            field = key.split(".", 1)[1]
            backtest[field] = value

    return ConfigBundle(market=market, strategy=strategy, backtest=backtest, risk=risk)


def _run_walk_forward(exp_dir: Path, base_bundle: ConfigBundle, scenario_name: str, cfg: dict) -> None:
    walk_cfg = cfg.get("walk_forward", {})
    windows = walk_cfg.get("windows", [])
    if not windows:
        return
    rows: list[dict[str, object]] = []
    for idx, window in enumerate(windows, start=1):
        bundle = ConfigBundle(
            market=dict(base_bundle.market),
            strategy=dict(base_bundle.strategy),
            backtest=deep_merge(dict(base_bundle.backtest), {"start": window["start"], "end": window["end"]}),
            risk=dict(base_bundle.risk),
        )
        provider = LocalCSVDataProvider(bundle.backtest.get("data_dir", "sample_data"))
        artifacts = BacktestEngine(bundle).run(provider)
        rows.append(
            {
                "window_id": idx,
                "start": window["start"],
                "end": window["end"],
                "total_return": artifacts.metrics.get("total_return", 0.0),
                "sharpe": artifacts.metrics.get("sharpe", 0.0),
            }
        )
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(exp_dir / f"{scenario_name}_walk_forward.csv", index=False)


def _run_regimes(exp_dir: Path, base_bundle: ConfigBundle, scenario_name: str, cfg: dict) -> None:
    regimes = cfg.get("regimes", [])
    if not regimes:
        return
    rows: list[dict[str, object]] = []
    provider = LocalCSVDataProvider(base_bundle.backtest.get("data_dir", "sample_data"))
    for regime in regimes:
        bundle = ConfigBundle(
            market=dict(base_bundle.market),
            strategy=dict(base_bundle.strategy),
            backtest=deep_merge(
                dict(base_bundle.backtest),
                {"start": regime["start"], "end": regime["end"]},
            ),
            risk=dict(base_bundle.risk),
        )
        artifacts = BacktestEngine(bundle).run(provider)
        rows.append(
            {
                "regime": regime.get("name", f"{regime['start']}_{regime['end']}"),
                "start": regime["start"],
                "end": regime["end"],
                "total_return": artifacts.metrics.get("total_return", 0.0),
                "annualized_return": artifacts.metrics.get("annualized_return", 0.0),
                "sharpe": artifacts.metrics.get("sharpe", 0.0),
            }
        )
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(exp_dir / f"{scenario_name}_regimes.csv", index=False)
