from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ConfigBundle:
    market: dict[str, Any]
    strategy: dict[str, Any]
    backtest: dict[str, Any]
    risk: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "strategy": self.strategy,
            "backtest": self.backtest,
            "risk": self.risk,
        }


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_ref(path_like: str | None, config_path: Path) -> dict[str, Any]:
    if not path_like:
        return {}
    ref = Path(path_like)
    if not ref.is_absolute():
        ref = (config_path.parent / ref).resolve()
    return load_yaml(ref)


def load_config_bundle(backtest_config_path: str | Path) -> ConfigBundle:
    backtest_path = Path(backtest_config_path).resolve()
    backtest = load_yaml(backtest_path)
    market = _resolve_ref(backtest.get("market_config"), backtest_path)
    strategy = _resolve_ref(backtest.get("strategy_config"), backtest_path)
    risk = _resolve_ref(backtest.get("risk_config"), backtest_path)

    market = deep_merge(market, backtest.get("market_overrides", {}))
    strategy = deep_merge(strategy, backtest.get("strategy_overrides", {}))
    risk = deep_merge(risk, backtest.get("risk_overrides", {}))

    return ConfigBundle(market=market, strategy=strategy, backtest=backtest, risk=risk)
