from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass(slots=True)
class IncrementalWindow:
    should_sync: bool
    sync_start: str
    reason: str


def resolve_incremental_window(
    output_dir: str | Path,
    end: str,
    default_start: str,
) -> IncrementalWindow:
    output = Path(output_dir)
    end_ts = pd.Timestamp(end).normalize()
    price_file = output / "price_history.csv"
    if not price_file.exists():
        return IncrementalWindow(should_sync=True, sync_start=default_start, reason="no_existing_dataset")

    try:
        frame = pd.read_csv(price_file, usecols=["date"])
    except Exception:
        return IncrementalWindow(should_sync=True, sync_start=default_start, reason="price_read_failed")
    if frame.empty:
        return IncrementalWindow(should_sync=True, sync_start=default_start, reason="empty_price_history")
    max_date = pd.to_datetime(frame["date"], errors="coerce").max()
    if pd.isna(max_date):
        return IncrementalWindow(should_sync=True, sync_start=default_start, reason="invalid_max_date")
    sync_start_ts = pd.Timestamp(max_date).normalize() + pd.Timedelta(days=1)
    if sync_start_ts > end_ts:
        return IncrementalWindow(should_sync=False, sync_start=end, reason="already_up_to_date")
    return IncrementalWindow(
        should_sync=True,
        sync_start=sync_start_ts.strftime("%Y-%m-%d"),
        reason=f"resume_from_{pd.Timestamp(max_date).strftime('%Y-%m-%d')}",
    )


def merge_sync_outputs(base_dir: str | Path, new_dir: str | Path) -> None:
    base = Path(base_dir)
    new = Path(new_dir)
    base.mkdir(parents=True, exist_ok=True)
    _merge_csv(
        base / "security_master.csv",
        new / "security_master.csv",
        dedup_keys=["symbol"],
    )
    _merge_csv(
        base / "price_history.csv",
        new / "price_history.csv",
        dedup_keys=["date", "symbol"],
        sort_keys=["date", "symbol"],
    )
    _merge_csv(
        base / "financials.csv",
        new / "financials.csv",
        dedup_keys=["symbol", "period_end"],
        sort_keys=["symbol", "period_end"],
    )
    _merge_csv(
        base / "release_calendar.csv",
        new / "release_calendar.csv",
        dedup_keys=["symbol", "period_end"],
        sort_keys=["symbol", "period_end"],
    )


def update_sync_checkpoint(
    output_dir: str | Path,
    market: str,
    start: str,
    end: str,
    reason: str,
    status: str,
    error: str | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "sync_checkpoint.json"
    payload = {}
    if checkpoint.exists():
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload.update(
        {
            "market": market,
            "last_start": start,
            "last_end": end,
            "last_reason": reason,
            "last_status": status,
            "last_error": error or "",
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        }
    )
    checkpoint.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _merge_csv(
    base_file: Path,
    new_file: Path,
    dedup_keys: list[str],
    sort_keys: list[str] | None = None,
) -> None:
    if not new_file.exists():
        return
    new_df = pd.read_csv(new_file)
    if base_file.exists():
        base_df = pd.read_csv(base_file)
        merged = pd.concat([base_df, new_df], ignore_index=True)
    else:
        merged = new_df
    dedup_subset = [key for key in dedup_keys if key in merged.columns]
    if dedup_subset:
        merged = merged.drop_duplicates(subset=dedup_subset, keep="last")
    if sort_keys:
        by = [key for key in sort_keys if key in merged.columns]
        if by:
            merged = merged.sort_values(by)
    merged.to_csv(base_file, index=False)
