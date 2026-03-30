from __future__ import annotations

from pathlib import Path

from hk_stock_quant.data.akshare_cn import AkshareCNDataProvider as LegacyCNProvider

from .local_csv import LocalCSVDataProvider


class AkshareCNDataProvider(LocalCSVDataProvider):
    """Sync A-share market data to local standardized CSV and reuse LocalCSV provider APIs."""

    @classmethod
    def sync_to_local_dataset(
        cls,
        output_dir: str | Path,
        start: str,
        end: str,
        symbols: list[str] | None = None,
        max_symbols: int | None = 500,
        benchmark_symbol: str = "sh000300",
        sleep_seconds: float = 0.1,
    ) -> Path:
        return LegacyCNProvider.sync_to_local_dataset(
            output_dir=output_dir,
            start=start,
            end=end,
            symbols=symbols,
            max_symbols=max_symbols,
            benchmark_symbol=benchmark_symbol,
            sleep_seconds=sleep_seconds,
        )
