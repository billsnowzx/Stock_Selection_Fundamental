"""Config-driven fundamental research framework for HK/CN equities."""

from .backtest import BacktestEngine
from .config import ConfigBundle, load_config_bundle, load_yaml

__all__ = ["BacktestEngine", "ConfigBundle", "load_yaml", "load_config_bundle"]
