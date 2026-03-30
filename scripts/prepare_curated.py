from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_selection_fundamental.config import load_config_bundle
from stock_selection_fundamental.providers import LocalCSVDataProvider, prepare_curated_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare curated snapshot dataset.")
    parser.add_argument("--config", default="configs/backtests/hk_top20.yaml")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    bundle = load_config_bundle(args.config)
    if args.data_dir:
        bundle.backtest["data_dir"] = args.data_dir
    source = Path(bundle.backtest.get("data_dir", "sample_data"))
    output = Path(args.output_dir) if args.output_dir else Path("data/curated") / str(bundle.backtest.get("name", "curated"))

    provider = LocalCSVDataProvider(source)
    result = prepare_curated_dataset(provider=provider, config_bundle=bundle, output_dir=output)
    print(f"Curated dataset prepared at: {result.output_dir.resolve()}")
    print(json.dumps(result.manifest, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
