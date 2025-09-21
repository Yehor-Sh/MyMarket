"""Inside bar breakout strategy implementation."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import ema


class InsideBarBreakoutStrategy(ModuleBase):
    """Identify inside bar breakouts confirmed by trend and volume."""

    def __init__(self, client, *, interval: str = "1h", lookback: int = 100) -> None:
        super().__init__(
            client,
            name="InsideBarBreakout",
            abbreviation="INS",
            interval=interval,
            lookback=max(lookback, 80),
        )
        self._ema_period = 20
        self._volume_window = 20

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < 3:
            return []

        mother, inside, breakout = candles[-3], candles[-2], candles[-1]
        if not (inside.high <= mother.high and inside.low >= mother.low):
            return []

        closes = [c.close for c in candles]
        ema_values = ema(closes, self._ema_period)
        if len(ema_values) < 2 or math.isnan(ema_values[-1]) or math.isnan(ema_values[-2]):
            return []

        if len(candles) <= self._volume_window:
            return []
        volume_slice = candles[-(self._volume_window + 1) : -1]
        avg_volume = mean(c.volume for c in volume_slice)
        if avg_volume <= 0:
            return []

        signals: List[Signal] = []
        metadata = {
            "mother_high": mother.high,
            "mother_low": mother.low,
            "inside_high": inside.high,
            "inside_low": inside.low,
            "breakout_volume": breakout.volume,
            "avg_volume": avg_volume,
        }

        trend_up = ema_values[-1] > ema_values[-2]
        trend_down = ema_values[-1] < ema_values[-2]
        volume_ok = breakout.volume >= avg_volume * 1.5

        if breakout.close > mother.high and trend_up and volume_ok:
            signals.append(
                self.make_signal(
                    symbol,
                    "LONG",
                    confidence=1.05,
                    metadata={
                        **metadata,
                        "direction": "up",
                        "ema_current": ema_values[-1],
                        "ema_previous": ema_values[-2],
                        "volume_multiple": breakout.volume / avg_volume if avg_volume else math.nan,
                    },
                )
            )
        if breakout.close < mother.low and trend_down and volume_ok:
            signals.append(
                self.make_signal(
                    symbol,
                    "SHORT",
                    confidence=1.05,
                    metadata={
                        **metadata,
                        "direction": "down",
                        "ema_current": ema_values[-1],
                        "ema_previous": ema_values[-2],
                        "volume_multiple": breakout.volume / avg_volume if avg_volume else math.nan,
                    },
                )
            )

        return signals


__all__ = ["InsideBarBreakoutStrategy"]
