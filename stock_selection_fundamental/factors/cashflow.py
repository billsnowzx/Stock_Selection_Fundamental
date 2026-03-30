from __future__ import annotations

from .base import safe_ratio


def fcf_conversion(row) -> float:
    direct = row.get("fcf_conversion")
    if direct is not None and not (direct != direct):
        return float(direct)
    return safe_ratio(row.get("free_cashflow"), row.get("net_income"))
