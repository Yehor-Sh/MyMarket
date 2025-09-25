"""N-day high/low breakout strategy."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import atr, ema


class BreakoutHighLowStrategy(ModuleBase):
    """Emit signals when price breaks out of recent ranges with volatility."""

    def __init__(
        self,
        client,
        *,
        interval: str = "4h",
        lookback: int = 200,
        breakout_period: int = 20,
        atr_period: int = 14,
        atr_median_period: int = 50,
        ema_fast_period: int = 20,
        ema_slow_period: int = 50,
        volume_window: int = 20,
    ) -> None:
        minimum_history = max(
            breakout_period + 1,
            atr_period + atr_median_period,
            ema_slow_period + 2,
            volume_window + 2,
        )
        super().__init__(
            client,
            name="BreakoutHighLow",
            abbreviation="BRK",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._breakout_period = breakout_period
        self._atr_period = atr_period
        self._atr_median_period = atr_median_period
        self._ema_fast_period = ema_fast_period
        self._ema_slow_period = ema_slow_period
        self._volume_window = volume_window

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self._breakout_period + 1:
            return []

        breakout_candle = candles[-1]
        high_window = candles[-(self._breakout_period + 1) : -1]
        recent_high = max(c.high for c in high_window)
        recent_low = min(c.low for c in high_window)

        atr_values = atr(candles, self._atr_period)
        if not atr_values or math.isnan(atr_values[-1]):
            return []
        atr_slice = [
            value
            for value in atr_values[-self._atr_median_period :]
            if not math.isnan(value)
        ]
        if not atr_slice:
            return []
        median_atr = median(atr_slice)
        current_atr = atr_values[-1]
        if current_atr <= median_atr:
            return []

        closes = [c.close for c in candles]
        ema_fast_values = ema(closes, self._ema_fast_period)
        ema_slow_values = ema(closes, self._ema_slow_period)
        if (
            not ema_fast_values
            or not ema_slow_values
            or math.isnan(ema_fast_values[-1])
            or math.isnan(ema_slow_values[-1])
        ):
            return []
        ema_fast_curr = ema_fast_values[-1]
        ema_slow_curr = ema_slow_values[-1]

        if len(candles) <= self._volume_window:
            return []
        volume_slice = candles[-(self._volume_window + 1) : -1]
        avg_volume = mean(c.volume for c in volume_slice)
        if avg_volume <= 0 or breakout_candle.volume <= avg_volume:
            return []

        signals: List[Signal] = []

        if (
            breakout_candle.close > recent_high
            and ema_fast_curr > ema_slow_curr
        ):
            metadata = {
                "breakout_level": recent_high,
                "atr": current_atr,
                "median_atr": median_atr,
                "direction": "up",
                "ema_fast": ema_fast_curr,
                "ema_slow": ema_slow_curr,
                "volume": breakout_candle.volume,
                "avg_volume": avg_volume,
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=1.1, metadata=metadata))

        if (
            breakout_candle.close < recent_low
            and ema_fast_curr < ema_slow_curr
        ):
            metadata = {
                "breakout_level": recent_low,
                "atr": current_atr,
                "median_atr": median_atr,
                "direction": "down",
                "ema_fast": ema_fast_curr,
                "ema_slow": ema_slow_curr,
                "volume": breakout_candle.volume,
                "avg_volume": avg_volume,
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=1.1, metadata=metadata))

        return signals


__all__ = ["BreakoutHighLowStrategy"]
