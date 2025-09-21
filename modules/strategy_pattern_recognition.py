"""Basic candlestick pattern recognition strategy."""

from __future__ import annotations

from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class PatternRecognitionStrategy(ModuleBase):
    """Look for simple two-candle reversal formations on 15 minute bars."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 40,
    ) -> None:
        super().__init__(
            client,
            name="PatternRecognition",
            abbreviation="PAT",
            interval=interval,
            lookback=max(lookback, 10),
        )

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < 2:
            return []

        prev = candles[-2]
        curr = candles[-1]
        signals: List[Signal] = []

        bullish_setup = (
            prev.close < prev.open
            and curr.close > curr.open
            and curr.close > prev.close
        )
        if bullish_setup:
            metadata = {
                "prev_open": prev.open,
                "prev_close": prev.close,
                "curr_open": curr.open,
                "curr_close": curr.close,
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=0.85, metadata=metadata))

        bearish_setup = (
            prev.close > prev.open
            and curr.close < curr.open
            and curr.close < prev.close
        )
        if bearish_setup:
            metadata = {
                "prev_open": prev.open,
                "prev_close": prev.close,
                "curr_open": curr.open,
                "curr_close": curr.close,
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.85, metadata=metadata))

        return signals


__all__ = ["PatternRecognitionStrategy"]
