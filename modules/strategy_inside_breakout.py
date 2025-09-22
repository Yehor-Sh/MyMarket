"""Inside bar breakout strategy with volume confirmation (INS)."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import ema


class InsideBarVolumeBreakoutStrategy(ModuleBase):
    """Breakouts following an inside bar pattern with trend confirmation."""

    def __init__(
        self,
        client,
        *,
        interval: str = "1h",
        lookback: int = 160,
        ema_fast_period: int = 20,
        ema_slow_period: int = 50,
        volume_window: int = 20,
        breakout_period: int = 10,
    ) -> None:
        minimum_history = max(
            ema_slow_period + 3,
            volume_window + 3,
            breakout_period + 3,
        )
        super().__init__(
            client,
            name="Inside Bar Breakout + Volume",
            abbreviation="INS",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._ema_fast_period = ema_fast_period
        self._ema_slow_period = ema_slow_period
        self._volume_window = volume_window
        self._breakout_period = breakout_period

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback or len(candles) < 3:
            return []

        mother = candles[-3]
        inside = candles[-2]
        breakout = candles[-1]

        if not (
            inside.high <= mother.high and inside.low >= mother.low
        ):
            return []

        closes = [candle.close for candle in candles]
        ema_fast = ema(closes, self._ema_fast_period)
        ema_slow = ema(closes, self._ema_slow_period)
        if (
            not ema_fast
            or not ema_slow
            or math.isnan(ema_fast[-1])
            or math.isnan(ema_slow[-1])
        ):
            return []
        ema_fast_current = ema_fast[-1]
        ema_slow_current = ema_slow[-1]

        volume_slice = candles[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        volume_ratio = breakout.volume / average_volume

        signals: List[Signal] = []

        recent_high = max(candle.high for candle in candles[-(self._breakout_period + 1) : -1])
        recent_low = min(candle.low for candle in candles[-(self._breakout_period + 1) : -1])

        if (
            breakout.close > mother.high
            and breakout.close > recent_high
            and ema_fast_current > ema_slow_current
            and volume_ratio >= 1.5
        ):
            metadata = {
                "mother_high": mother.high,
                "mother_low": mother.low,
                "ema_trend": "bullish",
                "volume_ratio": volume_ratio,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "LONG",
                    confidence=1.1,
                    metadata=metadata,
                )
            )

        if (
            breakout.close < mother.low
            and breakout.close < recent_low
            and ema_fast_current < ema_slow_current
            and volume_ratio >= 1.5
        ):
            metadata = {
                "mother_high": mother.high,
                "mother_low": mother.low,
                "ema_trend": "bearish",
                "volume_ratio": volume_ratio,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "SHORT",
                    confidence=1.1,
                    metadata=metadata,
                )
            )

        return signals


__all__ = ["InsideBarVolumeBreakoutStrategy"]
