from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskModelConfig:
    industry_neutral: bool = False
    style_neutral: bool = False
    max_single_weight: float = 0.1
