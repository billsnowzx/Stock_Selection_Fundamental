from __future__ import annotations

import numpy as np
import pandas as pd


def compute_composite_score(
    frame: pd.DataFrame,
    weights: dict[str, float],
    min_factors_required: int,
) -> pd.DataFrame:
    output = frame.copy()
    score_cols = [f"{name}_score" for name in weights]
    for col in score_cols:
        if col not in output.columns:
            output[col] = np.nan

    weight_series = pd.Series({f"{name}_score": value for name, value in weights.items()}, dtype=float)
    available = output[score_cols].notna().mul(weight_series, axis=1)
    weighted = output[score_cols].mul(weight_series, axis=1)

    valid_count = output[score_cols].notna().sum(axis=1)
    denom = available.sum(axis=1).replace(0, np.nan)
    output["composite_score"] = (weighted.sum(axis=1) / denom).where(valid_count >= min_factors_required)
    output["valid_factor_count"] = valid_count
    return output
