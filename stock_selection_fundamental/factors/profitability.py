from __future__ import annotations

from .base import safe_ratio


def roic(row) -> float:
    direct = row.get("roic")
    if direct is not None and not (direct != direct):
        return float(direct)
    return safe_ratio(row.get("nopat"), row.get("invested_capital"))


def net_margin(row) -> float:
    direct = row.get("net_margin")
    if direct is not None and not (direct != direct):
        return float(direct)
    return safe_ratio(row.get("net_income"), row.get("revenue"))
