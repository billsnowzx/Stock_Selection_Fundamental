from __future__ import annotations

import pandas as pd


def estimate_basic_exposures(selection_history: pd.DataFrame) -> pd.DataFrame:
    if selection_history.empty:
        return pd.DataFrame(columns=["signal_date", "industry", "weight"])
    if "industry" not in selection_history.columns:
        return pd.DataFrame(columns=["signal_date", "industry", "weight"])
    grouped = (
        selection_history.groupby(["signal_date", "industry"])["target_weight"]
        .sum()
        .reset_index(name="weight")
    )
    return grouped
