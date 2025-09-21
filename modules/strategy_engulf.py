"""Engulfing pattern strategy with RSI confirmation."""

from __future__ import annotations

import math
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import rsi


class EngulfingStrategy(ModuleBase):
    """Detect bullish and bearish engulfing setups confirmed by RSI."""

    def __init__(self, client, *, interval: str = "1h", lookback: int = 80) -> None:
        super().__init__(
            client,
            name="Engulfing",
            abbreviation="ENG",
            interval=interval,
            lookback=max(lookback, 60),
        )
        self._rsi_period = 14

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < 2:
            return []

        prev, curr = candles[-2], candles[-1]
        closes = [c.close for c in candles]
        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return []
        current_rsi = rsi_values[-1]

        signals: List[Signal] = []

        bullish_body = curr.open <= prev.close and curr.close >= prev.open
        bearish_body = curr.open >= prev.close and curr.close <= prev.open

        if prev.is_bearish and curr.is_bullish and bullish_body:
            closes_beyond = curr.close > prev.high
            if closes_beyond and current_rsi < 30.0:
                metadata = {
                    "rsi": current_rsi,
                    "prev_high": prev.high,
                    "prev_low": prev.low,
                    "curr_open": curr.open,
                    "curr_close": curr.close,
                }
                signals.append(self.make_signal(symbol, "LONG", confidence=0.95, metadata=metadata))

        if prev.is_bullish and curr.is_bearish and bearish_body:
            closes_beyond = curr.close < prev.low
            if closes_beyond and current_rsi > 70.0:
                metadata = {
                    "rsi": current_rsi,
                    "prev_high": prev.high,
                    "prev_low": prev.low,
                    "curr_open": curr.open,
                    "curr_close": curr.close,
                }
                signals.append(self.make_signal(symbol, "SHORT", confidence=0.95, metadata=metadata))

        return signals


__all__ = ["EngulfingStrategy"]
