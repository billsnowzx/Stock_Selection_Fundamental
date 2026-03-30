from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import json
import re
from copy import deepcopy

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
    scenarios_dir = exp_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _expand_scenarios(cfg.get("grid", {}), cfg.get("scenarios", []))
    scenario_rows: list[dict[str, object]] = []
    walk_rows: list[pd.DataFrame] = []
    regime_rows: list[pd.DataFrame] = []

    for idx, scenario in enumerate(scenarios, start=1):
        scenario_name = _scenario_name(scenario, idx=idx)
        scenario_dir = scenarios_dir / _safe_name(scenario_name)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        bundle = _apply_scenario_overrides(base_bundle, scenario)
        metrics = _run_period(
            bundle=bundle,
            period_name="full_period",
            output_dir=scenario_dir / "full_period",
            experiment_id=experiment_id,
            scenario_name=scenario_name,
            export_artifacts=True,
        )

        walk_df = _run_walk_forward(
            scenario_dir=scenario_dir,
            base_bundle=bundle,
            scenario_name=scenario_name,
            cfg=cfg,
            experiment_id=experiment_id,
        )
        if not walk_df.empty:
            walk_rows.append(walk_df)

        regime_df = _run_regimes(
            scenario_dir=scenario_dir,
            base_bundle=bundle,
            scenario_name=scenario_name,
            cfg=cfg,
            experiment_id=experiment_id,
        )
        if not regime_df.empty:
            regime_rows.append(regime_df)

        row = {
            "scenario_name": scenario_name,
            "total_return": metrics.get("total_return", 0.0),
            "annualized_return": metrics.get("annualized_return", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "turnover": metrics.get("turnover", 0.0),
            "overrides": json.dumps(scenario, ensure_ascii=False, default=str),
        }
        row.update(_summarize_walk_forward(walk_df))
        row.update(_summarize_regimes(regime_df))
        scenario_rows.append(row)

        (scenario_dir / "scenario_overrides.json").write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (scenario_dir / "bundle_snapshot.json").write_text(
            json.dumps(bundle.as_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    summary = pd.DataFrame(scenario_rows)
    if not summary.empty:
        summary = _add_ranking(summary)
        summary = summary.sort_values(["rank_score", "sharpe", "annualized_return"], ascending=[True, False, False]).reset_index(drop=True)

    summary.to_csv(exp_dir / "experiment_summary.csv", index=False)
    summary.to_csv(exp_dir / "experiment_ranking.csv", index=False)
    if walk_rows:
        pd.concat(walk_rows, ignore_index=True).to_csv(exp_dir / "walk_forward_summary.csv", index=False)
    if regime_rows:
        pd.concat(regime_rows, ignore_index=True).to_csv(exp_dir / "regime_summary.csv", index=False)

    (exp_dir / "experiment_config_snapshot.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return ExperimentRunResult(experiment_id=experiment_id, output_dir=exp_dir, summary=summary)


def _run_walk_forward(
    scenario_dir: Path,
    base_bundle: ConfigBundle,
    scenario_name: str,
    cfg: dict,
    experiment_id: str,
) -> pd.DataFrame:
    walk_cfg = cfg.get("walk_forward", {}) or {}
    folds = walk_cfg.get("folds", []) or []
    windows = walk_cfg.get("windows", []) or []
    save_artifacts = bool(walk_cfg.get("save_artifacts", False))
    rows: list[dict[str, object]] = []

    walk_dir = scenario_dir / "walk_forward"
    walk_dir.mkdir(parents=True, exist_ok=True)

    for idx, fold in enumerate(folds, start=1):
        fold_name = str(fold.get("name") or f"fold_{idx:02d}")
        train_metrics = _run_period(
            bundle=_period_bundle(base_bundle, fold["train_start"], fold["train_end"]),
            period_name=f"{fold_name}_train",
            output_dir=walk_dir / _safe_name(f"{fold_name}_train"),
            experiment_id=experiment_id,
            scenario_name=scenario_name,
            export_artifacts=save_artifacts,
        )
        test_metrics = _run_period(
            bundle=_period_bundle(base_bundle, fold["test_start"], fold["test_end"]),
            period_name=f"{fold_name}_test",
            output_dir=walk_dir / _safe_name(f"{fold_name}_test"),
            experiment_id=experiment_id,
            scenario_name=scenario_name,
            export_artifacts=save_artifacts,
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "mode": "fold",
                "segment_name": fold_name,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "train_sharpe": train_metrics.get("sharpe", 0.0),
                "test_sharpe": test_metrics.get("sharpe", 0.0),
                "train_total_return": train_metrics.get("total_return", 0.0),
                "test_total_return": test_metrics.get("total_return", 0.0),
                "sharpe_gap_test_minus_train": test_metrics.get("sharpe", 0.0) - train_metrics.get("sharpe", 0.0),
            }
        )

    for idx, window in enumerate(windows, start=1):
        seg_name = str(window.get("name") or f"window_{idx:02d}")
        metrics = _run_period(
            bundle=_period_bundle(base_bundle, window["start"], window["end"]),
            period_name=f"{seg_name}_oos",
            output_dir=walk_dir / _safe_name(f"{seg_name}_oos"),
            experiment_id=experiment_id,
            scenario_name=scenario_name,
            export_artifacts=save_artifacts,
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "mode": "window",
                "segment_name": seg_name,
                "train_start": None,
                "train_end": None,
                "test_start": window["start"],
                "test_end": window["end"],
                "train_sharpe": None,
                "test_sharpe": metrics.get("sharpe", 0.0),
                "train_total_return": None,
                "test_total_return": metrics.get("total_return", 0.0),
                "sharpe_gap_test_minus_train": None,
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.to_csv(scenario_dir / "walk_forward_summary.csv", index=False)
    return out


def _run_regimes(
    scenario_dir: Path,
    base_bundle: ConfigBundle,
    scenario_name: str,
    cfg: dict,
    experiment_id: str,
) -> pd.DataFrame:
    regimes = cfg.get("regimes", []) or []
    if not regimes:
        return pd.DataFrame()

    regime_cfg = cfg.get("regime_config", {}) or {}
    save_artifacts = bool(regime_cfg.get("save_artifacts", False))
    regime_dir = scenario_dir / "regimes"
    regime_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for idx, regime in enumerate(regimes, start=1):
        regime_name = str(regime.get("name") or f"regime_{idx:02d}")
        metrics = _run_period(
            bundle=_period_bundle(base_bundle, regime["start"], regime["end"]),
            period_name=regime_name,
            output_dir=regime_dir / _safe_name(regime_name),
            experiment_id=experiment_id,
            scenario_name=scenario_name,
            export_artifacts=save_artifacts,
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "regime_name": regime_name,
                "start": regime["start"],
                "end": regime["end"],
                "total_return": metrics.get("total_return", 0.0),
                "annualized_return": metrics.get("annualized_return", 0.0),
                "sharpe": metrics.get("sharpe", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(scenario_dir / "regime_summary.csv", index=False)
    return out


def _run_period(
    bundle: ConfigBundle,
    period_name: str,
    output_dir: Path,
    experiment_id: str,
    scenario_name: str,
    export_artifacts: bool,
) -> dict[str, float]:
    provider = LocalCSVDataProvider(bundle.backtest.get("data_dir", "sample_data"))
    artifacts = BacktestEngine(bundle).run(provider)
    if export_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_metadata = {
            "experiment_id": experiment_id,
            "scenario_name": scenario_name,
            "period_name": period_name,
            "config_hash": hash_config_bundle(bundle),
            "data_hash": hash_data_dir(bundle.backtest.get("data_dir", "sample_data")),
        }
        export_csv_outputs(
            artifacts=artifacts,
            output_dir=output_dir,
            config_snapshot=bundle.as_dict(),
            run_metadata=run_metadata,
            write_parquet=bool(bundle.backtest.get("storage", {}).get("write_parquet", False)),
        )
        export_html_report(artifacts=artifacts, output_dir=output_dir, config_snapshot=bundle.as_dict())
    return {k: float(v) for k, v in artifacts.metrics.items()}


def _period_bundle(base_bundle: ConfigBundle, start: str, end: str) -> ConfigBundle:
    return ConfigBundle(
        market=deepcopy(base_bundle.market),
        strategy=deepcopy(base_bundle.strategy),
        backtest=deep_merge(deepcopy(base_bundle.backtest), {"start": start, "end": end}),
        risk=deepcopy(base_bundle.risk),
    )


def _summarize_walk_forward(walk_df: pd.DataFrame) -> dict[str, float | int]:
    if walk_df.empty:
        return {}
    test_sharpe = pd.to_numeric(walk_df["test_sharpe"], errors="coerce").dropna()
    test_return = pd.to_numeric(walk_df["test_total_return"], errors="coerce").dropna()
    gap = pd.to_numeric(walk_df["sharpe_gap_test_minus_train"], errors="coerce").dropna()
    return {
        "walk_forward_segments": int(len(walk_df)),
        "walk_forward_test_sharpe_mean": float(test_sharpe.mean()) if not test_sharpe.empty else 0.0,
        "walk_forward_test_return_mean": float(test_return.mean()) if not test_return.empty else 0.0,
        "walk_forward_sharpe_gap_mean": float(gap.mean()) if not gap.empty else 0.0,
    }


def _summarize_regimes(regime_df: pd.DataFrame) -> dict[str, float | int]:
    if regime_df.empty:
        return {}
    sharpe = pd.to_numeric(regime_df["sharpe"], errors="coerce").dropna()
    returns = pd.to_numeric(regime_df["total_return"], errors="coerce").dropna()
    return {
        "regime_segments": int(len(regime_df)),
        "regime_sharpe_mean": float(sharpe.mean()) if not sharpe.empty else 0.0,
        "regime_return_mean": float(returns.mean()) if not returns.empty else 0.0,
        "regime_sharpe_min": float(sharpe.min()) if not sharpe.empty else 0.0,
    }


def _add_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    for col, asc in [
        ("sharpe", False),
        ("annualized_return", False),
        ("total_return", False),
        ("max_drawdown", False),
        ("turnover", True),
    ]:
        if col in ranked.columns:
            ranked[f"rank_{col}"] = ranked[col].rank(ascending=asc, method="min")
    score_cols = [c for c in ranked.columns if c.startswith("rank_")]
    if score_cols:
        ranked["rank_score"] = ranked[score_cols].mean(axis=1)
    else:
        ranked["rank_score"] = float("nan")
    ranked["rank_final"] = ranked["rank_score"].rank(ascending=True, method="min")
    return ranked


def _expand_scenarios(grid: dict[str, list[object]], scenarios: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    expanded: list[dict[str, object]] = []
    if grid:
        keys = sorted(grid.keys())
        values = [grid[key] if isinstance(grid[key], list) else [grid[key]] for key in keys]
        for combo in product(*values):
            scenario = {key: value for key, value in zip(keys, combo)}
            scenario["name"] = "__".join(f"{key}-{value}" for key, value in scenario.items() if key != "name")
            expanded.append(scenario)
    if scenarios:
        expanded.extend(scenarios)
    if not expanded:
        return [{"name": "default"}]
    return expanded


def _apply_scenario_overrides(base: ConfigBundle, scenario: dict[str, object]) -> ConfigBundle:
    market = deepcopy(base.market)
    strategy = deepcopy(base.strategy)
    backtest = deepcopy(base.backtest)
    risk = deepcopy(base.risk)

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
        elif key.startswith("portfolio."):
            field = key.split(".", 1)[1]
            strategy.setdefault("portfolio", {})
            strategy["portfolio"][field] = value
        elif key.startswith("risk."):
            field = key.split(".", 1)[1]
            risk[field] = value
        elif key.startswith("backtest."):
            field = key.split(".", 1)[1]
            backtest[field] = value
        elif key.startswith("market."):
            field = key.split(".", 1)[1]
            market[field] = value

    return ConfigBundle(market=market, strategy=strategy, backtest=backtest, risk=risk)


def _scenario_name(scenario: dict[str, object], idx: int) -> str:
    raw = str(scenario.get("name") or f"scenario_{idx:03d}")
    return raw if raw else f"scenario_{idx:03d}"


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "scenario"
