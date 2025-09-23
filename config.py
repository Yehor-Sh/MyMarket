"""Project configuration with runtime mode selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "mode": "live",  # "live" or "backtest"
    "backtest": {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "strategy": "ENG",
        "csv_path": None,  # Path to optional CSV with historical klines
        "limit": 1000,
        "trailing_percent": 2.0,
        "enable_trailing": True,
        "initial_capital": 10_000.0,
        "commission_pct": 0.0,
        "report_directory": str(Path("backtest_reports")),
    },
}

__all__ = ["CONFIG"]
