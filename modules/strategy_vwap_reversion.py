from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_K


class VwapReversionStrategy(LegacySnapshotStrategy):
    """VWAP reversion squeeze wrapper."""

    def __init__(self, client) -> None:
        lookback = max(module_K.VWAP_LEN + 3, 62)
        super().__init__(
            client,
            name="VWAP Reversion",
            abbreviation="VWAP",
            interval=module_K.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_K.detect,
        )


__all__ = ["VwapReversionStrategy"]
