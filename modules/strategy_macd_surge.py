from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_J


class MacdSurgeStrategy(LegacySnapshotStrategy):
    """MACD histogram surge wrapper."""

    def __init__(self, client) -> None:
        lookback = max(module_J.EMA_SLOW + module_J.EMA_SIG + 6, 62)
        super().__init__(
            client,
            name="MACD Surge",
            abbreviation="MACD",
            interval=module_J.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_J.detect,
        )


__all__ = ["MacdSurgeStrategy"]
