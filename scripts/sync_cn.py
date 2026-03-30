from __future__ import annotations

import argparse

from stock_selection_fundamental.providers.akshare_cn import AkshareCNDataProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CN dataset from AkShare.")
    parser.add_argument("--output-dir", default="data/curated/cn")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--max-symbols", type=int, default=500)
    parser.add_argument("--benchmark-symbol", default="sh000300")
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--full-sync", action="store_true")
    args = parser.parse_args()

    symbol_list = args.symbols if args.symbols else None
    output = AkshareCNDataProvider.sync_to_local_dataset(
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        symbols=symbol_list,
        max_symbols=None if symbol_list else args.max_symbols,
        benchmark_symbol=args.benchmark_symbol,
        sleep_seconds=args.sleep_seconds,
        incremental=not args.full_sync,
    )
    print(f"CN dataset synced to {output.resolve()}")


if __name__ == "__main__":
    main()
