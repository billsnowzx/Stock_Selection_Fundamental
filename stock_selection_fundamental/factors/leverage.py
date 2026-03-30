from __future__ import annotations

from .base import safe_ratio


def debt_to_cashflow(row) -> float:
    direct = row.get("debt_to_cashflow")
    if direct is not None and not (direct != direct):
        return float(direct)
    return safe_ratio(row.get("total_liabilities"), row.get("operating_cashflow"))
