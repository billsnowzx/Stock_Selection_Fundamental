from __future__ import annotations

import math

import pandas as pd


def rank_and_select(
    frame: pd.DataFrame,
    top_n: int | None,
    top_percentile: float | None,
    min_selection: int,
) -> pd.DataFrame:
    output = frame.dropna(subset=["composite_score"]).sort_values("composite_score", ascending=False).copy()
    output["rank"] = range(1, len(output) + 1)

    target_n = top_n
    if target_n is None and top_percentile is not None:
        target_n = max(min_selection, math.ceil(len(output) * top_percentile))
    if target_n is None:
        target_n = min_selection
    target_n = max(min_selection, int(target_n))
    return output.head(target_n).reset_index(drop=True)
