from __future__ import annotations

import pandas as pd


def compute_stability_bundle(
    scored_snapshots: dict[pd.Timestamp, pd.DataFrame],
    factor_names: list[str],
) -> dict[str, pd.DataFrame]:
    coverage_rows: list[dict[str, float | str | pd.Timestamp]] = []
    moments_rows: list[dict[str, float | str | pd.Timestamp]] = []
    panel: list[pd.DataFrame] = []

    for signal_date, snapshot in scored_snapshots.items():
        if snapshot.empty:
            continue
        frame = snapshot.copy()
        frame["signal_date"] = signal_date
        panel.append(frame[["signal_date", *[c for c in factor_names if c in frame.columns]]])
        for factor in factor_names:
            if factor not in snapshot.columns:
                continue
            values = snapshot[factor]
            coverage_rows.append(
                {
                    "signal_date": signal_date,
                    "factor": factor,
                    "coverage": float(values.notna().mean()),
                }
            )
            moments_rows.append(
                {
                    "signal_date": signal_date,
                    "factor": factor,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
            )

    coverage = pd.DataFrame(coverage_rows)
    moments = pd.DataFrame(moments_rows)

    if panel:
        combined = pd.concat(panel, ignore_index=True)
        corr = combined[[c for c in factor_names if c in combined.columns]].corr()
        corr = corr.reset_index().rename(columns={"index": "factor"})
    else:
        corr = pd.DataFrame()

    return {"coverage": coverage, "moments": moments, "correlation": corr}
