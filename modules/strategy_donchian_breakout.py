from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_F


class DonchianBreakoutStrategy(LegacySnapshotStrategy):
    """Expose the Donchian breakout logic via the unified interface."""

    def __init__(self, client) -> None:
        lookback = module_F.LOOKBACK + 2
        super().__init__(
            client,
            name="Donchian Breakout",
            abbreviation="DON",
            interval=module_F.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_F.detect,
        )


__all__ = ["DonchianBreakoutStrategy"]
