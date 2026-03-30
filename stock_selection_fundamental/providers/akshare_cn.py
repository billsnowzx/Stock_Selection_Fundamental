from __future__ import annotations

from pathlib import Path
import tempfile
import time

import pandas as pd

from hk_stock_quant.data.akshare_cn import AkshareCNDataProvider as LegacyCNProvider

from .local_csv import LocalCSVDataProvider
from .sync_utils import (
    load_sync_checkpoint,
    merge_sync_outputs,
    resolve_incremental_window,
    resolve_symbol_incremental_window,
    save_sync_checkpoint,
    update_symbol_checkpoint,
    update_sync_checkpoint,
)


class AkshareCNDataProvider(LocalCSVDataProvider):
    """Sync CN market data with symbol-level incremental checkpoint and resume."""

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
        incremental: bool = True,
        retry_times: int = 3,
        continue_on_error: bool = True,
    ) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = load_sync_checkpoint(out)
        checkpoint.update(
            {
                "market": "CN",
                "last_requested_start": start,
                "last_requested_end": end,
                "benchmark_symbol": benchmark_symbol,
            }
        )
        save_sync_checkpoint(out, checkpoint)

        target_symbols = cls._resolve_target_symbols(symbols=symbols, max_symbols=max_symbols)
        if not target_symbols:
            return cls._fallback_batch_sync(
                output_dir=out,
                start=start,
                end=end,
                symbols=symbols,
                max_symbols=max_symbols,
                benchmark_symbol=benchmark_symbol,
                sleep_seconds=sleep_seconds,
                incremental=incremental,
            )
        success_count = 0
        failed_count = 0

        for symbol in target_symbols:
            state = checkpoint.get("symbols", {}).get(symbol)
            if state is None and incremental:
                state = cls._infer_symbol_state_from_prices(out, symbol)
            window = resolve_symbol_incremental_window(state, requested_start=start, requested_end=end) if incremental else None
            if window is not None and not window.should_sync:
                checkpoint = update_symbol_checkpoint(
                    checkpoint,
                    symbol,
                    status="skipped",
                    start=window.sync_start,
                    end=end,
                    attempts=0,
                    reason=window.reason,
                    error="",
                )
                save_sync_checkpoint(out, checkpoint)
                continue
            sync_start = window.sync_start if window else start
            reason = window.reason if window else "full_symbol_sync"

            succeeded = False
            last_error = ""
            for attempt in range(1, max(int(retry_times), 1) + 1):
                try:
                    with tempfile.TemporaryDirectory(prefix="cn_sync_symbol_") as tmp:
                        tmp_path = Path(tmp)
                        LegacyCNProvider.sync_to_local_dataset(
                            output_dir=tmp_path,
                            start=sync_start,
                            end=end,
                            symbols=[symbol],
                            max_symbols=None,
                            benchmark_symbol=benchmark_symbol,
                            sleep_seconds=sleep_seconds,
                        )
                        merge_sync_outputs(base_dir=out, new_dir=tmp_path)
                    checkpoint = update_symbol_checkpoint(
                        checkpoint,
                        symbol,
                        status="success",
                        start=sync_start,
                        end=end,
                        attempts=attempt,
                        reason=reason,
                        error="",
                    )
                    save_sync_checkpoint(out, checkpoint)
                    success_count += 1
                    succeeded = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    checkpoint = update_symbol_checkpoint(
                        checkpoint,
                        symbol,
                        status="retrying" if attempt < retry_times else "failed",
                        start=sync_start,
                        end=end,
                        attempts=attempt,
                        reason=reason,
                        error=last_error,
                    )
                    save_sync_checkpoint(out, checkpoint)
                    if attempt < retry_times:
                        time.sleep(min(1.0 * attempt, 5.0))
            if not succeeded:
                failed_count += 1
                if not continue_on_error:
                    update_sync_checkpoint(
                        output_dir=out,
                        market="CN",
                        start=start,
                        end=end,
                        reason="stopped_on_first_error",
                        status="failed",
                        error=last_error,
                    )
                    raise RuntimeError(f"CN sync failed for {symbol}: {last_error}")

        final_status = "success" if failed_count == 0 else "partial_success"
        update_sync_checkpoint(
            output_dir=out,
            market="CN",
            start=start,
            end=end,
            reason=f"symbols={len(target_symbols)} success={success_count} failed={failed_count}",
            status=final_status,
            error="" if failed_count == 0 else "some_symbols_failed",
        )
        return out

    @classmethod
    def _resolve_target_symbols(cls, symbols: list[str] | None, max_symbols: int | None) -> list[str]:
        if symbols:
            return list(dict.fromkeys(str(s).strip() for s in symbols if str(s).strip()))
        try:
            master = LegacyCNProvider._fetch_security_master(symbols=None, max_symbols=max_symbols)
            if not master.empty and "symbol" in master.columns:
                return master["symbol"].dropna().astype(str).tolist()
        except Exception:
            pass
        return []

    @classmethod
    def _fallback_batch_sync(
        cls,
        *,
        output_dir: Path,
        start: str,
        end: str,
        symbols: list[str] | None,
        max_symbols: int | None,
        benchmark_symbol: str,
        sleep_seconds: float,
        incremental: bool,
    ) -> Path:
        window = resolve_incremental_window(output_dir=output_dir, end=end, default_start=start) if incremental else None
        if window is not None and not window.should_sync:
            update_sync_checkpoint(
                output_dir=output_dir,
                market="CN",
                start=start,
                end=end,
                reason=window.reason,
                status="skipped_up_to_date",
            )
            return output_dir
        sync_start = window.sync_start if window else start
        with tempfile.TemporaryDirectory(prefix="cn_sync_batch_") as tmp:
            tmp_path = Path(tmp)
            LegacyCNProvider.sync_to_local_dataset(
                output_dir=tmp_path,
                start=sync_start,
                end=end,
                symbols=symbols,
                max_symbols=max_symbols,
                benchmark_symbol=benchmark_symbol,
                sleep_seconds=sleep_seconds,
            )
            merge_sync_outputs(base_dir=output_dir, new_dir=tmp_path)
        update_sync_checkpoint(
            output_dir=output_dir,
            market="CN",
            start=sync_start,
            end=end,
            reason="fallback_batch_sync",
            status="success",
        )
        return output_dir

    @staticmethod
    def _infer_symbol_state_from_prices(output_dir: Path, symbol: str) -> dict[str, str] | None:
        price_path = output_dir / "price_history.csv"
        if not price_path.exists():
            return None
        try:
            frame = pd.read_csv(price_path)
        except Exception:
            return None
        if frame.empty or "symbol" not in frame.columns or "date" not in frame.columns:
            return None
        symbol_frame = frame.loc[frame["symbol"].astype(str) == str(symbol)].copy()
        if symbol_frame.empty:
            return None
        symbol_frame["date"] = pd.to_datetime(symbol_frame["date"], errors="coerce")
        last_date = symbol_frame["date"].dropna().max()
        if pd.isna(last_date):
            return None
        return {
            "status": "success",
            "last_end": pd.Timestamp(last_date).strftime("%Y-%m-%d"),
        }
