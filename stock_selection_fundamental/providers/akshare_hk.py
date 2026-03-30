from __future__ import annotations

from pathlib import Path
import tempfile

from hk_stock_quant.data.akshare_hk import AkshareHKDataProvider as LegacyHKProvider

from .local_csv import LocalCSVDataProvider
from .sync_utils import merge_sync_outputs, resolve_incremental_window, update_sync_checkpoint


class AkshareHKDataProvider(LocalCSVDataProvider):
    """Sync HK market data to local standardized CSV and reuse LocalCSV provider APIs."""

    @classmethod
    def sync_to_local_dataset(
        cls,
        output_dir: str | Path,
        start: str,
        end: str,
        symbols: list[str] | None = None,
        max_symbols: int | None = 300,
        benchmark_symbol: str = "HSI",
        sleep_seconds: float = 0.2,
        incremental: bool = True,
    ) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        window = resolve_incremental_window(output_dir=out, end=end, default_start=start) if incremental else None
        sync_start = window.sync_start if window else start
        if window is not None and not window.should_sync:
            update_sync_checkpoint(
                output_dir=out,
                market="HK",
                start=start,
                end=end,
                reason=window.reason,
                status="skipped_up_to_date",
            )
            return out

        try:
            with tempfile.TemporaryDirectory(prefix="hk_sync_tmp_") as tmp:
                tmp_path = Path(tmp)
                LegacyHKProvider.sync_to_local_dataset(
                    output_dir=tmp_path,
                    start=sync_start,
                    end=end,
                    symbols=symbols,
                    max_symbols=max_symbols,
                    benchmark_symbol=benchmark_symbol,
                    sleep_seconds=sleep_seconds,
                )
                merge_sync_outputs(base_dir=out, new_dir=tmp_path)
            update_sync_checkpoint(
                output_dir=out,
                market="HK",
                start=sync_start,
                end=end,
                reason=window.reason if window else "full_sync",
                status="success",
            )
            return out
        except Exception as exc:
            update_sync_checkpoint(
                output_dir=out,
                market="HK",
                start=sync_start,
                end=end,
                reason=window.reason if window else "full_sync",
                status="failed",
                error=str(exc),
            )
            raise
