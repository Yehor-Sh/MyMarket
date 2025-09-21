"""N-day high/low breakout strategy."""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import atr


class BreakoutHighLowStrategy(ModuleBase):
    """Emit signals when price breaks out of recent ranges with volatility."""

    def __init__(
        self,
        client,
        *,
        interval: str = "4h",
        lookback: int = 160,
        breakout_period: int = 20,
        atr_period: int = 14,
    ) -> None:
        super().__init__(
            client,
            name="BreakoutHighLow",
            abbreviation="BRK",
            interval=interval,
            lookback=max(lookback, breakout_period + atr_period + 10),
        )
        self._breakout_period = breakout_period
        self._atr_period = atr_period
        self._median_window = max(breakout_period, atr_period)

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
        atr_slice = [value for value in atr_values[-self._median_window :] if not math.isnan(value)]
        if not atr_slice:
            return []
        median_atr = median(atr_slice)
        current_atr = atr_values[-1]
        if current_atr <= median_atr:
            return []

        signals: List[Signal] = []

        if breakout_candle.close > recent_high:
            metadata = {
                "breakout_level": recent_high,
                "atr": current_atr,
                "median_atr": median_atr,
                "direction": "up",
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=1.1, metadata=metadata))

        if breakout_candle.close < recent_low:
            metadata = {
                "breakout_level": recent_low,
                "atr": current_atr,
                "median_atr": median_atr,
                "direction": "down",
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=1.1, metadata=metadata))

        return signals


__all__ = ["BreakoutHighLowStrategy"]
