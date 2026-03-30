from __future__ import annotations

import pandas as pd


def optimize_weights(initial: pd.Series, risk_penalty: float = 0.0) -> pd.Series:
    """Placeholder optimizer interface for second stage upgrades."""
    if initial.empty:
        return initial
    weights = initial.clip(lower=0)
    total = weights.sum()
    return weights / total if total > 0 else weights
