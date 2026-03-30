from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
import json
import re
import time

import pandas as pd

from ..backtest.engine import BacktestEngine
from ..config import ConfigBundle, deep_merge, load_config_bundle, load_yaml
from ..providers.local_csv import LocalCSVDataProvider
from ..reporting.export_csv import export_csv_outputs
from ..reporting.export_html import export_html_report
from ..runtime import (
    append_run_audit,
    generate_run_id,
    hash_config_bundle,
    hash_data_dir,
    utc_now_iso,
)


@dataclass(slots=True)
class ExperimentRunResult:
    experiment_id: str
    output_dir: Path
    summary: pd.DataFrame


@dataclass(slots=True)
class PeriodRunResult:
    metrics: dict[str, float]
    source: str
    status: str
    error: str
    duration_sec: float
    config_hash: str
    data_hash: str
    run_state_path: Path


def run_experiment_suite(
    config_path: str | Path,
    resume: bool | None = None,
    fail_fast: bool | None = None,
) -> ExperimentRunResult:
    cfg_path = Path(config_path).resolve()
    cfg = load_yaml(cfg_path)
    base_backtest_cfg = cfg.get("base_backtest_config", "configs/backtests/hk_top20.yaml")
    base_bundle = load_config_bundle(base_backtest_cfg)

    resume_run = bool(cfg.get("resume", True)) if resume is None else bool(resume)
    fail_fast_run = bool(cfg.get("fail_fast", False)) if fail_fast is None else bool(fail_fast)
    output_root = Path(cfg.get("output_root", "outputs/experiments"))
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_id = str(cfg.get("experiment_id") or generate_run_id(prefix="exp"))
    run_instance_id = generate_run_id(prefix="exp_run")
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

        scenario_started = time.perf_counter()
        scenario_status = "success"
        scenario_error = ""
        full_period = PeriodRunResult(
            metrics={},
            source="none",
            status="failed",
            error="",
            duration_sec=0.0,
            config_hash=hash_config_bundle(bundle),
            data_hash=hash_data_dir(bundle.backtest.get("data_dir", "sample_data")),
            run_state_path=scenario_dir / "full_period" / "run_state.json",
        )
        walk_df = pd.DataFrame()
        regime_df = pd.DataFrame()

        try:
            full_period = _run_period(
                bundle=bundle,
                period_name="full_period",
                output_dir=scenario_dir / "full_period",
                experiment_id=experiment_id,
                scenario_name=scenario_name,
                run_instance_id=run_instance_id,
                export_artifacts=True,
                resume=resume_run,
            )
            walk_df = _run_walk_forward(
                scenario_dir=scenario_dir,
                base_bundle=bundle,
                scenario_name=scenario_name,
                cfg=cfg,
                experiment_id=experiment_id,
                run_instance_id=run_instance_id,
                resume=resume_run,
                fail_fast=fail_fast_run,
            )
            regime_df = _run_regimes(
                scenario_dir=scenario_dir,
                base_bundle=bundle,
                scenario_name=scenario_name,
                cfg=cfg,
                experiment_id=experiment_id,
                run_instance_id=run_instance_id,
                resume=resume_run,
                fail_fast=fail_fast_run,
            )
        except Exception as exc:
            scenario_status = "failed"
            scenario_error = str(exc)
            if fail_fast_run:
                raise

        if not walk_df.empty:
            walk_rows.append(walk_df)
        if not regime_df.empty:
            regime_rows.append(regime_df)

        row = {
            "scenario_name": scenario_name,
            "scenario_status": scenario_status,
            "scenario_error": scenario_error,
            "total_return": full_period.metrics.get("total_return", 0.0),
            "annualized_return": full_period.metrics.get("annualized_return", 0.0),
            "sharpe": full_period.metrics.get("sharpe", 0.0),
            "max_drawdown": full_period.metrics.get("max_drawdown", 0.0),
            "turnover": full_period.metrics.get("turnover", 0.0),
            "full_period_source": full_period.source,
            "full_period_status": full_period.status,
            "full_period_duration_sec": full_period.duration_sec,
            "config_hash": full_period.config_hash,
            "data_hash": full_period.data_hash,
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

        append_run_audit(
            exp_dir,
            {
                "kind": "experiment_scenario",
                "experiment_id": experiment_id,
                "run_instance_id": run_instance_id,
                "scenario_name": scenario_name,
                "status": scenario_status,
                "error": scenario_error,
                "duration_sec": time.perf_counter() - scenario_started,
                "full_period_source": full_period.source,
                "config_hash": full_period.config_hash,
                "data_hash": full_period.data_hash,
            },
            filename="experiment_runs.jsonl",
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

    config_snapshot = dict(cfg)
    config_snapshot["resolved"] = {
        "resume": resume_run,
        "fail_fast": fail_fast_run,
        "run_instance_id": run_instance_id,
    }
    (exp_dir / "experiment_config_snapshot.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    append_run_audit(
        output_root,
        {
            "kind": "experiment_suite",
            "experiment_id": experiment_id,
            "run_instance_id": run_instance_id,
            "scenario_count": int(len(scenario_rows)),
            "resume": resume_run,
            "fail_fast": fail_fast_run,
            "output_dir": str(exp_dir.resolve()),
        },
        filename="run_audit.jsonl",
    )
    return ExperimentRunResult(experiment_id=experiment_id, output_dir=exp_dir, summary=summary)


def _run_walk_forward(
    scenario_dir: Path,
    base_bundle: ConfigBundle,
    scenario_name: str,
    cfg: dict[str, Any],
    experiment_id: str,
    run_instance_id: str,
    resume: bool,
    fail_fast: bool,
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
        try:
            train_res = _run_period(
                bundle=_period_bundle(base_bundle, fold["train_start"], fold["train_end"]),
                period_name=f"{fold_name}_train",
                output_dir=walk_dir / _safe_name(f"{fold_name}_train"),
                experiment_id=experiment_id,
                scenario_name=scenario_name,
                run_instance_id=run_instance_id,
                export_artifacts=save_artifacts,
                resume=resume,
            )
            test_res = _run_period(
                bundle=_period_bundle(base_bundle, fold["test_start"], fold["test_end"]),
                period_name=f"{fold_name}_test",
                output_dir=walk_dir / _safe_name(f"{fold_name}_test"),
                experiment_id=experiment_id,
                scenario_name=scenario_name,
                run_instance_id=run_instance_id,
                export_artifacts=save_artifacts,
                resume=resume,
            )
            row_status = "success"
            row_error = ""
        except Exception as exc:
            if fail_fast:
                raise
            train_res = PeriodRunResult({}, "none", "failed", str(exc), 0.0, "", "", walk_dir / "run_state.json")
            test_res = PeriodRunResult({}, "none", "failed", str(exc), 0.0, "", "", walk_dir / "run_state.json")
            row_status = "failed"
            row_error = str(exc)

        rows.append(
            {
                "scenario_name": scenario_name,
                "mode": "fold",
                "segment_name": fold_name,
                "segment_status": row_status,
                "segment_error": row_error,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "train_source": train_res.source,
                "test_source": test_res.source,
                "train_sharpe": train_res.metrics.get("sharpe", 0.0),
                "test_sharpe": test_res.metrics.get("sharpe", 0.0),
                "train_total_return": train_res.metrics.get("total_return", 0.0),
                "test_total_return": test_res.metrics.get("total_return", 0.0),
                "sharpe_gap_test_minus_train": test_res.metrics.get("sharpe", 0.0) - train_res.metrics.get("sharpe", 0.0),
            }
        )

    for idx, window in enumerate(windows, start=1):
        seg_name = str(window.get("name") or f"window_{idx:02d}")
        try:
            test_res = _run_period(
                bundle=_period_bundle(base_bundle, window["start"], window["end"]),
                period_name=f"{seg_name}_oos",
                output_dir=walk_dir / _safe_name(f"{seg_name}_oos"),
                experiment_id=experiment_id,
                scenario_name=scenario_name,
                run_instance_id=run_instance_id,
                export_artifacts=save_artifacts,
                resume=resume,
            )
            row_status = "success"
            row_error = ""
        except Exception as exc:
            if fail_fast:
                raise
            test_res = PeriodRunResult({}, "none", "failed", str(exc), 0.0, "", "", walk_dir / "run_state.json")
            row_status = "failed"
            row_error = str(exc)

        rows.append(
            {
                "scenario_name": scenario_name,
                "mode": "window",
                "segment_name": seg_name,
                "segment_status": row_status,
                "segment_error": row_error,
                "train_start": None,
                "train_end": None,
                "test_start": window["start"],
                "test_end": window["end"],
                "train_source": "n/a",
                "test_source": test_res.source,
                "train_sharpe": None,
                "test_sharpe": test_res.metrics.get("sharpe", 0.0),
                "train_total_return": None,
                "test_total_return": test_res.metrics.get("total_return", 0.0),
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
    cfg: dict[str, Any],
    experiment_id: str,
    run_instance_id: str,
    resume: bool,
    fail_fast: bool,
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
        try:
            period = _run_period(
                bundle=_period_bundle(base_bundle, regime["start"], regime["end"]),
                period_name=regime_name,
                output_dir=regime_dir / _safe_name(regime_name),
                experiment_id=experiment_id,
                scenario_name=scenario_name,
                run_instance_id=run_instance_id,
                export_artifacts=save_artifacts,
                resume=resume,
            )
            row_status = "success"
            row_error = ""
        except Exception as exc:
            if fail_fast:
                raise
            period = PeriodRunResult({}, "none", "failed", str(exc), 0.0, "", "", regime_dir / "run_state.json")
            row_status = "failed"
            row_error = str(exc)

        rows.append(
            {
                "scenario_name": scenario_name,
                "regime_name": regime_name,
                "regime_status": row_status,
                "regime_error": row_error,
                "source": period.source,
                "start": regime["start"],
                "end": regime["end"],
                "total_return": period.metrics.get("total_return", 0.0),
                "annualized_return": period.metrics.get("annualized_return", 0.0),
                "sharpe": period.metrics.get("sharpe", 0.0),
                "max_drawdown": period.metrics.get("max_drawdown", 0.0),
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
    run_instance_id: str,
    export_artifacts: bool,
    resume: bool,
) -> PeriodRunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    t0 = time.perf_counter()

    data_dir = Path(bundle.backtest.get("data_dir", "sample_data"))
    config_hash = hash_config_bundle(bundle)
    data_hash = hash_data_dir(data_dir)
    run_state_path = output_dir / "run_state.json"
    metrics_path = output_dir / "metrics.json"

    if resume:
        cached = _load_cached_period(run_state_path, metrics_path, config_hash=config_hash, data_hash=data_hash)
        if cached is not None:
            return PeriodRunResult(
                metrics=cached,
                source="cached",
                status="success",
                error="",
                duration_sec=0.0,
                config_hash=config_hash,
                data_hash=data_hash,
                run_state_path=run_state_path,
            )

    try:
        provider = LocalCSVDataProvider(data_dir)
        artifacts = BacktestEngine(bundle).run(provider)
        metrics = {k: float(v) for k, v in artifacts.metrics.items()}

        if export_artifacts:
            run_metadata = {
                "experiment_id": experiment_id,
                "scenario_name": scenario_name,
                "period_name": period_name,
                "run_instance_id": run_instance_id,
                "config_hash": config_hash,
                "data_hash": data_hash,
            }
            export_csv_outputs(
                artifacts=artifacts,
                output_dir=output_dir,
                config_snapshot=bundle.as_dict(),
                run_metadata=run_metadata,
                write_parquet=bool(bundle.backtest.get("storage", {}).get("write_parquet", False)),
            )
            export_html_report(artifacts=artifacts, output_dir=output_dir, config_snapshot=bundle.as_dict())

        ended_at = utc_now_iso()
        duration_sec = time.perf_counter() - t0
        state_payload = {
            "status": "success",
            "experiment_id": experiment_id,
            "scenario_name": scenario_name,
            "period_name": period_name,
            "run_instance_id": run_instance_id,
            "source": "executed",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_sec": duration_sec,
            "config_hash": config_hash,
            "data_hash": data_hash,
            "metrics": metrics,
            "export_artifacts": export_artifacts,
        }
        run_state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return PeriodRunResult(
            metrics=metrics,
            source="executed",
            status="success",
            error="",
            duration_sec=duration_sec,
            config_hash=config_hash,
            data_hash=data_hash,
            run_state_path=run_state_path,
        )
    except Exception as exc:
        ended_at = utc_now_iso()
        duration_sec = time.perf_counter() - t0
        state_payload = {
            "status": "failed",
            "experiment_id": experiment_id,
            "scenario_name": scenario_name,
            "period_name": period_name,
            "run_instance_id": run_instance_id,
            "source": "executed",
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_sec": duration_sec,
            "config_hash": config_hash,
            "data_hash": data_hash,
            "error": str(exc),
            "export_artifacts": export_artifacts,
        }
        run_state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        raise


def _load_cached_period(
    run_state_path: Path,
    metrics_path: Path,
    *,
    config_hash: str,
    data_hash: str,
) -> dict[str, float] | None:
    if not run_state_path.exists():
        return None
    try:
        payload = json.loads(run_state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("status") != "success":
        return None
    if payload.get("config_hash") != config_hash or payload.get("data_hash") != data_hash:
        return None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(metrics, dict):
                return {k: float(v) for k, v in metrics.items()}
        except Exception:
            pass
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        try:
            return {k: float(v) for k, v in metrics.items()}
        except Exception:
            return None
    return None


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


def _expand_scenarios(
    grid: dict[str, list[object]],
    scenarios: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
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
