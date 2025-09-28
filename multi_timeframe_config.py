"""Configuration describing multi-timeframe requirements for modules."""
from __future__ import annotations
from typing import Dict

MultiTimeframeConfig = Dict[str, Dict[str, int]]

MULTI_TIMEFRAME_CONFIG: MultiTimeframeConfig = {
    "RMT": {
        "5m": 50,
        "15m": 50,
        "30m": 50,
    },
    "ATS": {
        "1h": 100,
    },
    "ENG": {
        "30m": 120,
        "1h": 120,
    },
}

__all__ = ["MULTI_TIMEFRAME_CONFIG", "MultiTimeframeConfig"]
