from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hk_stock_quant.demo_data import write_demo_dataset

from .backtest.engine import BacktestEngine
from .config import load_config_bundle
from .providers import AkshareCNDataProvider, AkshareHKDataProvider, LocalCSVDataProvider
from .reporting import export_csv_outputs, export_html_report


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

    sync_cn = subparsers.add_parser("sync-akshare-cn", help="Sync CN market dataset with AkShare.")
    sync_cn.add_argument("--output-dir", default="data/curated/cn")
    sync_cn.add_argument("--start", default="2020-01-01")
    sync_cn.add_argument("--end", default="2025-12-31")
    sync_cn.add_argument("--symbols", nargs="*", default=[])
    sync_cn.add_argument("--max-symbols", type=int, default=500)
    sync_cn.add_argument("--benchmark-symbol", default="sh000300")
    sync_cn.add_argument("--sleep-seconds", type=float, default=0.1)

    run = subparsers.add_parser("run-backtest", help="Run config-driven backtest.")
    run.add_argument("--config", default="configs/backtests/hk_top20.yaml")
    run.add_argument("--data-dir", default="")
    run.add_argument("--output-dir", default="")
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _run_backtest(config_path: str, data_dir_override: str, output_dir_override: str) -> None:
    bundle = load_config_bundle(config_path)
    if data_dir_override:
        bundle.backtest["data_dir"] = data_dir_override
    if output_dir_override:
        bundle.backtest["output_dir"] = output_dir_override

    provider_name = str(bundle.backtest.get("provider", "local_csv")).lower()
    data_dir = Path(bundle.backtest.get("data_dir", "sample_data"))
    if provider_name not in {"local_csv", "akshare_hk", "akshare_cn"}:
        raise ValueError(f"Unsupported provider for run-backtest: {provider_name}")
    provider = LocalCSVDataProvider(data_dir)

    logger.info("Running backtest with config: %s", Path(config_path).resolve())
    engine = BacktestEngine(config_bundle=bundle)
    artifacts = engine.run(provider)
    output_dir = bundle.backtest.get("output_dir", "outputs")
    output_path = export_csv_outputs(artifacts=artifacts, output_dir=output_dir, config_snapshot=bundle.as_dict())
    report_path = export_html_report(artifacts=artifacts, output_dir=output_path, config_snapshot=bundle.as_dict())
    print(json.dumps(artifacts.metrics, indent=2, ensure_ascii=False))
    print(f"CSV outputs: {output_path.resolve()}")
    print(f"HTML report: {report_path.resolve()}")


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
        )
        print(f"AkShare CN dataset synced to {output_path.resolve()}")
        return

    _run_backtest(
        config_path=args.config,
        data_dir_override=args.data_dir,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
