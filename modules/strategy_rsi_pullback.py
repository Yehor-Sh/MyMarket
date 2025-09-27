from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_H


class RsiPullbackStrategy(LegacySnapshotStrategy):
    """RSI pullback trend-following wrapper."""

    def __init__(self, client) -> None:
        lookback = max(module_H.RSI_LEN + 3, module_H.SMA_TREND + 2, 42)
        super().__init__(
            client,
            name="RSI Pullback",
            abbreviation="RSIP",
            interval=module_H.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_H.detect,
        )


__all__ = ["RsiPullbackStrategy"]
