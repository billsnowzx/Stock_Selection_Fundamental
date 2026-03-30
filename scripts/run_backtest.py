from __future__ import annotations

import argparse
import json

from stock_selection_fundamental.backtest.engine import BacktestEngine
from stock_selection_fundamental.config import load_config_bundle
from stock_selection_fundamental.providers.local_csv import LocalCSVDataProvider
from stock_selection_fundamental.reporting.export_csv import export_csv_outputs
from stock_selection_fundamental.reporting.export_html import export_html_report
from stock_selection_fundamental.runtime import generate_run_id, hash_config_bundle, hash_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a config-driven backtest.")
    parser.add_argument("--config", default="configs/backtests/hk_top20.yaml")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    bundle = load_config_bundle(args.config)
    if args.data_dir:
        bundle.backtest["data_dir"] = args.data_dir
    if args.output_dir:
        bundle.backtest["output_dir"] = args.output_dir

    run_id = args.run_id or generate_run_id(prefix="bt")
    provider = LocalCSVDataProvider(bundle.backtest.get("data_dir", "sample_data"))
    artifacts = BacktestEngine(bundle).run(provider)

    output_root = bundle.backtest.get("output_dir", "outputs")
    output_dir = f"{output_root}/{run_id}"
    metadata = {
        "run_id": run_id,
        "data_hash": hash_data_dir(bundle.backtest.get("data_dir", "sample_data")),
        "config_hash": hash_config_bundle(bundle),
    }
    output_path = export_csv_outputs(
        artifacts=artifacts,
        output_dir=output_dir,
        config_snapshot=bundle.as_dict(),
        run_metadata=metadata,
        write_parquet=bool(bundle.backtest.get("storage", {}).get("write_parquet", False)),
    )
    report_path = export_html_report(artifacts=artifacts, output_dir=output_path, config_snapshot=bundle.as_dict())
    print(json.dumps(artifacts.metrics, indent=2, ensure_ascii=False))
    print(f"CSV outputs: {output_path.resolve()}")
    print(f"HTML report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
