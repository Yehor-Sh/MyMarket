"""Volume weighted trend following strategy."""

from __future__ import annotations

from statistics import fmean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class VolumeWeightedTrendStrategy(ModuleBase):
    """Look for strong moves supported by a surge in traded volume."""

    def __init__(
        self,
        client,
        *,
        interval: str = "5m",
        lookback: int = 80,
        volume_window: int = 20,
        momentum_window: int = 5,
        volume_multiplier: float = 1.5,
    ) -> None:
        minimum_history = max(
            lookback,
            volume_window + 2,
            momentum_window + 2,
        )
        super().__init__(
            client,
            name="VolumeWeightedTrend",
            abbreviation="VWT",
            interval=interval,
            lookback=minimum_history,
        )
        self._volume_window = volume_window
        self._momentum_window = momentum_window
        self._volume_multiplier = volume_multiplier

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) <= max(self._volume_window, self._momentum_window):
            return []

        closes = [c.close for c in candles]
        last_close = closes[-1]
        momentum_reference = closes[-self._momentum_window]

        volume_slice = candles[-self._volume_window :]
        avg_volume = fmean(c.volume for c in volume_slice)
        current_volume = candles[-1].volume

        if avg_volume <= 0 or current_volume <= avg_volume * self._volume_multiplier:
            return []

        metadata = {
            "avg_volume": avg_volume,
            "current_volume": current_volume,
            "momentum_reference": momentum_reference,
            "last_close": last_close,
        }

        if last_close > momentum_reference:
            return [self.make_signal(symbol, "LONG", confidence=1.05, metadata=metadata)]

        if last_close < momentum_reference:
            return [self.make_signal(symbol, "SHORT", confidence=1.05, metadata=metadata)]

        return []


__all__ = ["VolumeWeightedTrendStrategy"]
