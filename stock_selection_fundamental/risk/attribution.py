from __future__ import annotations

import pandas as pd


def placeholder_return_attribution() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "market_component",
            "industry_component",
            "style_component",
            "selection_component",
        ]
    )
