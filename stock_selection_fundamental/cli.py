from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

from hk_stock_quant.demo_data import write_demo_dataset

from .backtest.engine import BacktestEngine
from .config import load_config_bundle
from .providers import (
    AkshareCNDataProvider,
    AkshareHKDataProvider,
    LocalCSVDataProvider,
    prepare_curated_dataset,
)
from .reporting import export_csv_outputs, export_html_report
from .research.experiments import run_experiment_suite
from .research.regression import compare_baseline_metrics, freeze_baseline_metrics
from .runtime import append_run_audit, generate_run_id, hash_config_bundle, hash_data_dir, utc_now_iso


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fundamental stock selection research framework (HK/CN).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-demo-data", help="Generate deterministic local demo CSV data.")
    generate.add_argument("--output-dir", default="sample_data")
    generate.add_argument("--start", default="2021-01-01")
    generate.add_argument("--end", default="2025-12-31")
    generate.add_argument("--symbols", type=int, default=30)

    sync_hk = subparsers.add_parser("sync-akshare-hk", help="Sync HK market dataset with AkShare.")
    sync_hk.add_argument("--output-dir", default="data/curated/hk")
    sync_hk.add_argument("--start", default="2020-01-01")
    sync_hk.add_argument("--end", default="2025-12-31")
    sync_hk.add_argument("--symbols", nargs="*", default=[])
    sync_hk.add_argument("--max-symbols", type=int, default=300)
    sync_hk.add_argument("--benchmark-symbol", default="HSI")
    sync_hk.add_argument("--sleep-seconds", type=float, default=0.2)
    sync_hk.add_argument("--full-sync", action="store_true", help="Disable incremental sync and rebuild from start/end.")
    sync_hk.add_argument("--retry-times", type=int, default=3)
    sync_hk.add_argument("--fail-fast", action="store_true", help="Stop immediately when a symbol sync fails.")

    sync_cn = subparsers.add_parser("sync-akshare-cn", help="Sync CN market dataset with AkShare.")
    sync_cn.add_argument("--output-dir", default="data/curated/cn")
    sync_cn.add_argument("--start", default="2020-01-01")
    sync_cn.add_argument("--end", default="2025-12-31")
    sync_cn.add_argument("--symbols", nargs="*", default=[])
    sync_cn.add_argument("--max-symbols", type=int, default=500)
    sync_cn.add_argument("--benchmark-symbol", default="sh000300")
    sync_cn.add_argument("--sleep-seconds", type=float, default=0.1)
    sync_cn.add_argument("--full-sync", action="store_true", help="Disable incremental sync and rebuild from start/end.")
    sync_cn.add_argument("--retry-times", type=int, default=3)
    sync_cn.add_argument("--fail-fast", action="store_true", help="Stop immediately when a symbol sync fails.")

    run = subparsers.add_parser("run-backtest", help="Run config-driven backtest.")
    run.add_argument("--config", default="configs/backtests/hk_top20.yaml")
    run.add_argument("--data-dir", default="")
    run.add_argument("--output-dir", default="")
    run.add_argument("--run-id", default="")

    curated = subparsers.add_parser("prepare-curated", help="Prepare curated dataset with visibility controls.")
    curated.add_argument("--config", default="configs/backtests/hk_top20.yaml")
    curated.add_argument("--data-dir", default="")
    curated.add_argument("--output-dir", default="")

    experiment = subparsers.add_parser("run-experiment", help="Run experiment suite with parameter grid.")
    experiment.add_argument("--config", default="configs/experiments/fundamental_grid.yaml")
    experiment.add_argument("--resume", dest="resume", action="store_true")
    experiment.add_argument("--no-resume", dest="resume", action="store_false")
    experiment.set_defaults(resume=None)
    experiment.add_argument("--fail-fast", dest="fail_fast", action="store_true")
    experiment.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    experiment.set_defaults(fail_fast=None)

    freeze = subparsers.add_parser("freeze-baseline", help="Freeze baseline metrics for regression checks.")
    freeze.add_argument("--metrics-file", default="outputs/hk_top20/metrics.json")
    freeze.add_argument("--output", default="tests/baseline/baseline_metrics.json")

    check = subparsers.add_parser("check-baseline", help="Compare current metrics against frozen baseline.")
    check.add_argument("--baseline", default="tests/baseline/baseline_metrics.json")
    check.add_argument("--metrics-file", default="outputs/hk_top20/metrics.json")
    check.add_argument("--tolerance-bps", type=float, default=200.0)
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _make_provider(provider_name: str, data_dir: Path):
    if provider_name not in {"local_csv", "akshare_hk", "akshare_cn"}:
        raise ValueError(f"Unsupported provider: {provider_name}")
    return LocalCSVDataProvider(data_dir)


def _run_backtest(config_path: str, data_dir_override: str, output_dir_override: str, run_id_override: str) -> dict[str, float]:
    bundle = load_config_bundle(config_path)
    if data_dir_override:
        bundle.backtest["data_dir"] = data_dir_override
    if output_dir_override:
        bundle.backtest["output_dir"] = output_dir_override

    base_output_dir = Path(bundle.backtest.get("output_dir", "outputs"))
    provider_name = str(bundle.backtest.get("provider", "local_csv")).lower()
    data_dir = Path(bundle.backtest.get("data_dir", "sample_data"))
    provider = _make_provider(provider_name, data_dir)
    run_id = run_id_override or generate_run_id(prefix="bt")
    output_dir = base_output_dir / run_id
    started_at = utc_now_iso()
    t0 = time.perf_counter()
    config_hash = hash_config_bundle(bundle)
    data_hash = hash_data_dir(data_dir)
    status = "success"
    error = ""

    logger.info("Running backtest with config=%s run_id=%s", Path(config_path).resolve(), run_id)
    try:
        engine = BacktestEngine(config_bundle=bundle)
        artifacts = engine.run(provider)
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        append_run_audit(
            base_output_dir,
            {
                "kind": "backtest",
                "run_id": run_id,
                "status": status,
                "error": error,
                "provider": provider_name,
                "config_path": str(Path(config_path).resolve()),
                "data_dir": str(data_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                "config_hash": config_hash,
                "data_hash": data_hash,
                "data_version": data_hash,
                "started_at": started_at,
                "ended_at": utc_now_iso(),
                "duration_sec": time.perf_counter() - t0,
            },
        )

    run_metadata = {
        "run_id": run_id,
        "provider": provider_name,
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "data_hash": data_hash,
        "config_hash": config_hash,
        "data_version": data_hash,
        "started_at": started_at,
        "ended_at": utc_now_iso(),
    }

    write_parquet = bool(bundle.backtest.get("storage", {}).get("write_parquet", False))
    output_path = export_csv_outputs(
        artifacts=artifacts,
        output_dir=output_dir,
        config_snapshot=bundle.as_dict(),
        run_metadata=run_metadata,
        write_parquet=write_parquet,
    )
    report_path = export_html_report(artifacts=artifacts, output_dir=output_path, config_snapshot=bundle.as_dict())
    print(json.dumps(artifacts.metrics, indent=2, ensure_ascii=False))
    print(f"CSV outputs: {output_path.resolve()}")
    print(f"HTML report: {report_path.resolve()}")
    return artifacts.metrics


def _prepare_curated(config_path: str, data_dir_override: str, output_dir_override: str) -> None:
    bundle = load_config_bundle(config_path)
    if data_dir_override:
        bundle.backtest["data_dir"] = data_dir_override
    source_dir = Path(bundle.backtest.get("data_dir", "sample_data"))
    provider = LocalCSVDataProvider(source_dir)

    output_dir = (
        Path(output_dir_override)
        if output_dir_override
        else Path("data/curated") / str(bundle.backtest.get("name", "curated"))
    )
    result = prepare_curated_dataset(provider=provider, config_bundle=bundle, output_dir=output_dir)
    print(f"Curated dataset prepared at: {result.output_dir.resolve()}")
    print(json.dumps(result.manifest, indent=2, ensure_ascii=False, default=str))


def _run_experiment(config_path: str, resume: bool | None, fail_fast: bool | None) -> None:
    result = run_experiment_suite(config_path, resume=resume, fail_fast=fail_fast)
    print(f"Experiment ID: {result.experiment_id}")
    print(f"Output Dir: {result.output_dir.resolve()}")
    print(result.summary.head(20).to_string(index=False))


def _freeze_baseline(metrics_file: str, output: str) -> None:
    metrics = json.loads(Path(metrics_file).read_text(encoding="utf-8"))
    path = freeze_baseline_metrics(metrics=metrics, output_path=output)
    print(f"Baseline frozen: {path.resolve()}")


def _check_baseline(baseline: str, metrics_file: str, tolerance_bps: float) -> None:
    metrics = json.loads(Path(metrics_file).read_text(encoding="utf-8"))
    result = compare_baseline_metrics(baseline_path=baseline, current_metrics=metrics, tolerance_bps=tolerance_bps)
    print(result.details.to_string(index=False))
    if not result.passed:
        raise SystemExit("Baseline regression check failed.")
    print("Baseline regression check passed.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    if args.command == "generate-demo-data":
        write_demo_dataset(
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            n_symbols=args.symbols,
        )
        print(f"Demo dataset generated in {Path(args.output_dir).resolve()}")
        return

    if args.command == "sync-akshare-hk":
        symbol_list = args.symbols if args.symbols else None
        output_path = AkshareHKDataProvider.sync_to_local_dataset(
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            symbols=symbol_list,
            max_symbols=None if symbol_list else args.max_symbols,
            benchmark_symbol=args.benchmark_symbol,
            sleep_seconds=args.sleep_seconds,
            incremental=not args.full_sync,
            retry_times=args.retry_times,
            continue_on_error=not args.fail_fast,
        )
        print(f"AkShare HK dataset synced to {output_path.resolve()}")
        return

    if args.command == "sync-akshare-cn":
        symbol_list = args.symbols if args.symbols else None
        output_path = AkshareCNDataProvider.sync_to_local_dataset(
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            symbols=symbol_list,
            max_symbols=None if symbol_list else args.max_symbols,
            benchmark_symbol=args.benchmark_symbol,
            sleep_seconds=args.sleep_seconds,
            incremental=not args.full_sync,
            retry_times=args.retry_times,
            continue_on_error=not args.fail_fast,
        )
        print(f"AkShare CN dataset synced to {output_path.resolve()}")
        return

    if args.command == "prepare-curated":
        _prepare_curated(args.config, args.data_dir, args.output_dir)
        return

    if args.command == "run-experiment":
        _run_experiment(args.config, args.resume, args.fail_fast)
        return

    if args.command == "freeze-baseline":
        _freeze_baseline(args.metrics_file, args.output)
        return

    if args.command == "check-baseline":
        _check_baseline(args.baseline, args.metrics_file, args.tolerance_bps)
        return

    _run_backtest(
        config_path=args.config,
        data_dir_override=args.data_dir,
        output_dir_override=args.output_dir,
        run_id_override=args.run_id,
    )


if __name__ == "__main__":
    main()
