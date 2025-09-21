"""Engulfing pattern strategy with RSI confirmation."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import atr, rsi


class EngulfingStrategy(ModuleBase):
    """Detect bullish and bearish engulfing setups confirmed by RSI."""

    def __init__(self, client, *, interval: str = "1h", lookback: int = 120) -> None:
        super().__init__(
            client,
            name="Engulfing",
            abbreviation="ENG",
            interval=interval,
            lookback=max(lookback, 80),
        )
        self._rsi_period = 14
        self._atr_period = 14
        self._atr_window = 20
        self._volume_window = 20

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < 2:
            return []

        prev, curr = candles[-2], candles[-1]
        closes = [c.close for c in candles]
        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return []
        current_rsi = rsi_values[-1]

        atr_values = atr(candles, self._atr_period)
        if not atr_values or math.isnan(atr_values[-1]):
            return []
        atr_sample = [value for value in atr_values[-self._atr_window :] if not math.isnan(value)]
        if not atr_sample:
            return []
        avg_atr = mean(atr_sample)
        current_atr = atr_values[-1]
        if current_atr <= avg_atr:
            return []

        if len(candles) <= self._volume_window:
            return []
        volume_slice = candles[-(self._volume_window + 1) : -1]
        avg_volume = mean(c.volume for c in volume_slice)
        if avg_volume <= 0:
            return []

        signals: List[Signal] = []

        bullish_body = curr.open <= prev.close and curr.close >= prev.open
        bearish_body = curr.open >= prev.close and curr.close <= prev.open

        if prev.is_bearish and curr.is_bullish and bullish_body:
            closes_beyond = curr.close > prev.high
            volume_ok = curr.volume >= avg_volume * 1.2
            if closes_beyond and volume_ok and current_rsi < 25.0:
                metadata = {
                    "rsi": current_rsi,
                    "prev_high": prev.high,
                    "prev_low": prev.low,
                    "curr_open": curr.open,
                    "curr_close": curr.close,
                    "volume": curr.volume,
                    "avg_volume": avg_volume,
                    "atr": current_atr,
                    "avg_atr": avg_atr,
                }
                signals.append(self.make_signal(symbol, "LONG", confidence=0.95, metadata=metadata))

        if prev.is_bullish and curr.is_bearish and bearish_body:
            closes_beyond = curr.close < prev.low
            volume_ok = curr.volume >= avg_volume * 1.2
            if closes_beyond and volume_ok and current_rsi > 75.0:
                metadata = {
                    "rsi": current_rsi,
                    "prev_high": prev.high,
                    "prev_low": prev.low,
                    "curr_open": curr.open,
                    "curr_close": curr.close,
                    "volume": curr.volume,
                    "avg_volume": avg_volume,
                    "atr": current_atr,
                    "avg_atr": avg_atr,
                }
                signals.append(self.make_signal(symbol, "SHORT", confidence=0.95, metadata=metadata))

        return signals


__all__ = ["EngulfingStrategy"]
