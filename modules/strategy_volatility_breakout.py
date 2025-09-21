"""ATR style volatility breakout strategy."""

from __future__ import annotations

from statistics import fmean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class VolatilityBreakoutStrategy(ModuleBase):
    """Trigger when price expands beyond the recent range by an ATR multiple."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 80,
        atr_period: int = 15,
        atr_multiplier: float = 1.0,
    ) -> None:
        minimum_history = max(lookback, atr_period + 2)
        super().__init__(
            client,
            name="VolatilityBreakout",
            abbreviation="VBO",
            interval=interval,
            lookback=minimum_history,
        )
        self._atr_period = atr_period
        self._atr_multiplier = atr_multiplier

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) <= self._atr_period:
            return []

        last = candles[-1]
        prev = candles[-2]
        atr_sample = candles[-self._atr_period :]
        ranges = [candle.high - candle.low for candle in atr_sample]
        atr_value = fmean(ranges)
        threshold = atr_value * self._atr_multiplier

        signals: List[Signal] = []
        price_delta = last.close - prev.close

        if price_delta > threshold:
            metadata = {
                "atr": atr_value,
                "previous_close": prev.close,
                "current_close": last.close,
                "change": price_delta,
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=1.0, metadata=metadata))

        if price_delta < -threshold:
            metadata = {
                "atr": atr_value,
                "previous_close": prev.close,
                "current_close": last.close,
                "change": price_delta,
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=1.0, metadata=metadata))

        return signals


__all__ = ["VolatilityBreakoutStrategy"]
