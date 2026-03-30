from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import pandas as pd

from ..config import ConfigBundle
from ..backtest.calendar import build_trading_dates, generate_signal_dates
from .base import DataProvider


@dataclass(slots=True)
class CuratedBuildResult:
    output_dir: Path
    manifest: dict[str, str | int | float]


def prepare_curated_dataset(
    provider: DataProvider,
    config_bundle: ConfigBundle,
    output_dir: str | Path,
    use_cache: bool = True,
) -> CuratedBuildResult:
    backtest_cfg = config_bundle.backtest
    start = pd.Timestamp(backtest_cfg["start"])
    end = pd.Timestamp(backtest_cfg["end"])
    benchmark_symbol = str(
        backtest_cfg.get("benchmark_symbol")
        or config_bundle.market.get("benchmark_symbol")
        or "^HSI"
    )
    frequency = str(
        backtest_cfg.get("rebalance_frequency")
        or config_bundle.strategy.get("signals", {}).get("rebalance_frequency")
        or "M"
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    master = provider.get_security_master().copy()
    prices = provider.get_price_history(start=start, end=end).copy()
    benchmark = provider.get_benchmark_history(benchmark_symbol, start=start, end=end).copy()
    if not benchmark.empty and prices[prices["symbol"] == benchmark_symbol].empty:
        prices = pd.concat([prices, benchmark], ignore_index=True)
    adjustment = provider.get_adjustment_factors(start=start, end=end)
    prices = _apply_adjustment_factors(prices, adjustment)

    releases = provider.get_release_calendar(start=start - pd.DateOffset(years=5), end=end).copy()
    financials = _load_full_financials(
        provider=provider,
        symbols=master["symbol"].tolist(),
        releases=releases,
    )
    lot_sizes = provider.get_lot_sizes(master["symbol"].tolist()).copy()
    actions = provider.get_corporate_actions(start=start, end=end).copy()
    industries = provider.get_industry_mapping(master["symbol"].tolist(), as_of_date=end).copy()

    source_hash = _hash_frames(
        {
            "security_master": master,
            "price_history": prices,
            "financials": financials,
            "release_calendar": releases,
            "adjustment_factors": adjustment,
            "corporate_actions": actions,
            "lot_sizes": lot_sizes,
            "industry_mapping": industries,
        }
    )
    manifest_path = output_path / "curated_manifest.json"
    if use_cache and manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached.get("source_hash") == source_hash:
            return CuratedBuildResult(output_dir=output_path, manifest=cached)

    prices = prices.sort_values(["date", "symbol"]).reset_index(drop=True)
    releases = releases.sort_values(["symbol", "release_date", "period_end"]).reset_index(drop=True)
    financials = financials.sort_values(["symbol", "period_end"]).reset_index(drop=True)
    visibility_snapshot = _build_visibility_snapshot(
        releases=releases,
        signal_dates=generate_signal_dates(
            build_trading_dates(prices, benchmark_symbol=benchmark_symbol),
            frequency=frequency,
        ),
    )

    master.to_csv(output_path / "security_master.csv", index=False)
    prices.to_csv(output_path / "price_history.csv", index=False)
    financials.to_csv(output_path / "financials.csv", index=False)
    releases.to_csv(output_path / "release_calendar.csv", index=False)
    lot_sizes.to_csv(output_path / "lot_size.csv", index=False)
    actions.to_csv(output_path / "corporate_actions.csv", index=False)
    industries.to_csv(output_path / "industry_mapping.csv", index=False)
    visibility_snapshot.to_csv(output_path / "financials_visibility_snapshot.csv", index=False)

    manifest = {
        "start": str(start.date()),
        "end": str(end.date()),
        "benchmark_symbol": benchmark_symbol,
        "frequency": frequency,
        "n_symbols": int(master["symbol"].nunique()) if "symbol" in master.columns else 0,
        "n_price_rows": int(len(prices)),
        "n_financial_rows": int(len(financials)),
        "source_hash": source_hash,
        "dataset_hash": _hash_frames(
            {
                "security_master": master,
                "price_history": prices,
                "financials": financials,
                "release_calendar": releases,
                "lot_size": lot_sizes,
                "industry_mapping": industries,
            }
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return CuratedBuildResult(output_dir=output_path, manifest=manifest)


def _apply_adjustment_factors(price_history: pd.DataFrame, adjustment_factors: pd.DataFrame) -> pd.DataFrame:
    if price_history.empty or adjustment_factors.empty:
        return price_history
    adjusted = price_history.copy()
    factor = adjustment_factors.copy()
    factor["date"] = pd.to_datetime(factor["date"], errors="coerce")
    factor["adj_factor"] = pd.to_numeric(factor["adj_factor"], errors="coerce").fillna(1.0)
    merged = adjusted.merge(factor[["date", "symbol", "adj_factor"]], on=["date", "symbol"], how="left")
    merged["adj_factor"] = merged["adj_factor"].fillna(1.0)
    for col in ("open", "high", "low", "close"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce") * merged["adj_factor"]
    return merged.drop(columns=["adj_factor"])


def _load_full_financials(
    provider: DataProvider,
    symbols: list[str],
    releases: pd.DataFrame,
) -> pd.DataFrame:
    history = provider.get_financial_history(symbols=symbols)
    if not history.empty:
        return history
    if releases.empty:
        return pd.DataFrame()
    max_release = releases["release_date"].max()
    return provider.get_financials(symbols=symbols, as_of_date=max_release)


def _build_visibility_snapshot(releases: pd.DataFrame, signal_dates: list[pd.Timestamp]) -> pd.DataFrame:
    if releases.empty or not signal_dates:
        return pd.DataFrame(columns=["signal_date", "symbol", "period_end", "release_date", "is_visible"])
    rows: list[dict[str, object]] = []
    for signal_date in signal_dates:
        visible = releases[releases["release_date"] <= signal_date].copy()
        if visible.empty:
            continue
        latest = visible.sort_values(["symbol", "release_date", "period_end"]).groupby("symbol").tail(1)
        latest["signal_date"] = signal_date
        latest["is_visible"] = True
        rows.extend(latest[["signal_date", "symbol", "period_end", "release_date", "is_visible"]].to_dict("records"))
    return pd.DataFrame(rows)


def _hash_frames(named_frames: dict[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for key in sorted(named_frames):
        frame = named_frames[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(len(frame)).encode("utf-8"))
        if frame.empty:
            continue
        csv_bytes = frame.sort_index(axis=1).to_csv(index=False).encode("utf-8")
        digest.update(csv_bytes)
    return digest.hexdigest()
