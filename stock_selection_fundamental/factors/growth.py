from __future__ import annotations


def revenue_growth_yoy(row) -> float:
    direct = row.get("revenue_growth_yoy")
    if direct is not None and not (direct != direct):
        return float(direct)
    prev = row.get("prev_revenue")
    curr = row.get("revenue")
    if prev is None or curr is None or prev <= 0 or curr <= 0:
        return float("nan")
    return (float(curr) - float(prev)) / float(prev)


def net_income_growth_yoy(row) -> float:
    direct = row.get("net_income_growth_yoy")
    if direct is not None and not (direct != direct):
        return float(direct)
    prev = row.get("prev_net_income")
    curr = row.get("net_income")
    if prev is None or curr is None or prev <= 0 or curr <= 0:
        return float("nan")
    return (float(curr) - float(prev)) / float(prev)
