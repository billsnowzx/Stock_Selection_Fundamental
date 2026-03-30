from __future__ import annotations

import json
from typing import Any

import pandas as pd


def metrics_to_frame(metrics: dict[str, float]) -> pd.DataFrame:
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    return pd.DataFrame(rows)


def dict_to_pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
