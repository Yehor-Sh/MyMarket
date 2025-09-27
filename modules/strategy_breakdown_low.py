from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_C


class BreakdownLowStrategy(LegacySnapshotStrategy):
    """Mirror the legacy breakdown low detector inside the tester."""

    def __init__(self, client) -> None:
        lookback = max(
            module_C.LOOKBACK_BARS + 2,
            module_C.SMA_LONG + 3,
            module_C.ATR_LEN + 3,
        )
        super().__init__(
            client,
            name="Breakdown Low",
            abbreviation="BRKL",
            interval=module_C.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_C.detect_signals,
        )


__all__ = ["BreakdownLowStrategy"]
