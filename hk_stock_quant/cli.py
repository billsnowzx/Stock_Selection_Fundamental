from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import BacktestEngine
from .config import StrategyConfig
from .data.akshare_cn import AkshareCNDataProvider
from .data.akshare_hk import AkshareHKDataProvider
from .data.local_csv import LocalCSVDataProvider
from .demo_data import write_demo_dataset
from .reporting import export_backtest_report
from .strategy import FundamentalTopNStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HK/A-share stock selection and backtesting framework.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-demo-data", help="Generate deterministic demo CSV data.")
    generate.add_argument("--output-dir", default="sample_data")
    generate.add_argument("--start", default="2021-01-01")
    generate.add_argument("--end", default="2025-12-31")
    generate.add_argument("--symbols", type=int, default=30)

    sync_hk = subparsers.add_parser("sync-akshare-hk", help="Sync real Hong Kong market data from AkShare into local CSV files.")
    sync_hk.add_argument("--output-dir", default="real_data_hk")
    sync_hk.add_argument("--start", default="2020-01-01")
    sync_hk.add_argument("--end", default="2025-12-31")
    sync_hk.add_argument("--symbols", default="")
    sync_hk.add_argument("--max-symbols", type=int, default=50)
    sync_hk.add_argument("--benchmark-symbol", default="HSI")
    sync_hk.add_argument("--sleep-seconds", type=float, default=0.2)

    sync_cn = subparsers.add_parser("sync-akshare-cn", help="Sync real A-share market data from AkShare into local CSV files.")
    sync_cn.add_argument("--output-dir", default="real_data_cn")
    sync_cn.add_argument("--start", default="2020-01-01")
    sync_cn.add_argument("--end", default="2025-12-31")
    sync_cn.add_argument("--symbols", default="")
    sync_cn.add_argument("--max-symbols", type=int, default=100)
    sync_cn.add_argument("--benchmark-symbol", default="sh000300")
    sync_cn.add_argument("--sleep-seconds", type=float, default=0.1)

    run = subparsers.add_parser("run-backtest", help="Run the monthly Top-N backtest on local CSV data.")
    run.add_argument("--data-dir", default="sample_data")
    run.add_argument("--output-dir", default="outputs")
    run.add_argument("--start", default="2023-01-03")
    run.add_argument("--end", default="2025-12-31")
    run.add_argument("--top-n", type=int, default=20)
    run.add_argument("--initial-capital", type=float, default=1000000.0)
    run.add_argument("--benchmark-symbol", default="^HSI")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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
        symbol_list = [item.strip() for item in args.symbols.split(",") if item.strip()] or None
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
        symbol_list = [item.strip() for item in args.symbols.split(",") if item.strip()] or None
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

    config = StrategyConfig(
        top_n=args.top_n,
        initial_capital=args.initial_capital,
        benchmark_symbol=args.benchmark_symbol,
    )
    provider = LocalCSVDataProvider(args.data_dir)
    strategy = FundamentalTopNStrategy(config)
    engine = BacktestEngine(config, strategy)
    result = engine.run(provider, start=args.start, end=args.end)
    output_path = export_backtest_report(result, args.output_dir, config)
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    print(f"Report exported to {output_path.resolve()}")


if __name__ == "__main__":
    main()
