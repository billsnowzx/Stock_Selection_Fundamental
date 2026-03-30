from __future__ import annotations

from pathlib import Path
import json

from ..types import BacktestArtifacts


def export_csv_outputs(
    artifacts: BacktestArtifacts,
    output_dir: str | Path,
    config_snapshot: dict,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts.nav_history.to_csv(output_path / "nav_history.csv", index=False)
    artifacts.trades.to_csv(output_path / "trades.csv", index=False)
    artifacts.holdings_history.to_csv(output_path / "holdings_history.csv", index=False)
    artifacts.selection_history.to_csv(output_path / "selection_history.csv", index=False)

    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts.metrics, handle, indent=2, ensure_ascii=False)
    with (output_path / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config_snapshot, handle, indent=2, ensure_ascii=False, default=str)

    for name, frame in artifacts.research_outputs.items():
        if frame is None:
            continue
        frame.to_csv(output_path / f"{name}.csv", index=False)
    return output_path
