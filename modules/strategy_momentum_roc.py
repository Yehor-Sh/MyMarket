from __future__ import annotations

from .legacy_adapter import LegacySnapshotStrategy
from . import module_E


class MomentumRocStrategy(LegacySnapshotStrategy):
    """Wrap the legacy ROC momentum detector."""

    def __init__(self, client) -> None:
        zscore_window = module_E.ZSCORE_WINDOW if module_E.ZSCORE_ENABLE else 0
        lookback = max(
            module_E.ROC_PERIOD + 2,
            module_E.SMA_LONG + 2,
            zscore_window + module_E.ROC_PERIOD + 1,
        )
        super().__init__(
            client,
            name="Momentum ROC",
            abbreviation="MROC",
            interval=module_E.INTERVAL_STRING,
            lookback=lookback,
            detect_func=module_E.detect_signals,
        )


__all__ = ["MomentumRocStrategy"]
