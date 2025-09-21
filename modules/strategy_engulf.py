"""Engulfing pattern strategy."""

from __future__ import annotations

from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class EngulfingStrategy(ModuleBase):
    """Detect bullish and bearish engulfing candlestick patterns."""

    def __init__(self, client, *, interval: str = "15m", lookback: int = 40) -> None:
        super().__init__(
            client,
            name="Engulfing",
            abbreviation="ENG",
            interval=interval,
            lookback=max(lookback, 10),
        )

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        prev, curr = candles[-2], candles[-1]
        signals: List[Signal] = []

        # Bullish engulfing: previous candle bearish, current candle bullish and
        # completely engulfs the body of the previous candle.
        if prev.is_bearish and curr.is_bullish:
            if curr.open <= prev.close and curr.close >= prev.open and curr.body > prev.body * 1.05:
                metadata = {
                    "prev_body": prev.body,
                    "curr_body": curr.body,
                    "upper_wick": curr.upper_wick,
                    "lower_wick": curr.lower_wick,
                }
                signals.append(self.make_signal(symbol, "LONG", confidence=1.0, metadata=metadata))

        # Bearish engulfing: inverse of the bullish setup.
        if prev.is_bullish and curr.is_bearish:
            if curr.open >= prev.close and curr.close <= prev.open and curr.body > prev.body * 1.05:
                metadata = {
                    "prev_body": prev.body,
                    "curr_body": curr.body,
                    "upper_wick": curr.upper_wick,
                    "lower_wick": curr.lower_wick,
                }
                signals.append(self.make_signal(symbol, "SHORT", confidence=1.0, metadata=metadata))

        return signals


__all__ = ["EngulfingStrategy"]
