from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_I


class BollingerSqueezeStrategy(LegacySnapshotStrategy):
    """Bollinger Band squeeze breakout wrapper."""

    def __init__(self, client) -> None:
        lookback = max(module_I.BB_LEN + 3, 62)
        super().__init__(
            client,
            name="Bollinger Squeeze",
            abbreviation="BBSQ",
            interval=module_I.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_I.detect,
        )


__all__ = ["BollingerSqueezeStrategy"]
