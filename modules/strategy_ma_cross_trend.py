from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_G


class MaCrossTrendStrategy(LegacySnapshotStrategy):
    """Expose the moving average cross trend detector."""

    def __init__(self, client) -> None:
        lookback = max(module_G.SMA_SLOW + 2, 50)
        super().__init__(
            client,
            name="MA Cross Trend",
            abbreviation="MACR",
            interval=module_G.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_G.detect,
        )


__all__ = ["MaCrossTrendStrategy"]
