from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_D


class RangeSurgeStrategy(LegacySnapshotStrategy):
    """Wrap the legacy range surge detector."""

    def __init__(self, client) -> None:
        lookback = max(
            module_D.LOOKBACK_BARS + 2,
            module_D.SMA_LONG + 3,
            module_D.ATR_LEN + 3,
        )
        super().__init__(
            client,
            name="Range Surge",
            abbreviation="RNGS",
            interval=module_D.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_D.detect_signals,
        )


__all__ = ["RangeSurgeStrategy"]
