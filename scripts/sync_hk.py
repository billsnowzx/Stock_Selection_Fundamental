from __future__ import annotations

import argparse

from stock_selection_fundamental.providers.akshare_hk import AkshareHKDataProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync HK dataset from AkShare.")
    parser.add_argument("--output-dir", default="data/curated/hk")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--max-symbols", type=int, default=300)
    parser.add_argument("--benchmark-symbol", default="HSI")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--full-sync", action="store_true")
    parser.add_argument("--retry-times", type=int, default=3)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    symbol_list = args.symbols if args.symbols else None
    output = AkshareHKDataProvider.sync_to_local_dataset(
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
    print(f"HK dataset synced to {output.resolve()}")


if __name__ == "__main__":
    main()
