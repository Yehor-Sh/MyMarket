"""Three white soldiers / three black crows strategy."""

from __future__ import annotations

from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class ThreeLineStrikeStrategy(ModuleBase):
    """Look for strong three candle continuation/reversal patterns."""

    def __init__(self, client, *, interval: str = "30m", lookback: int = 80) -> None:
        super().__init__(
            client,
            name="3Line",
            abbreviation="3WL",
            interval=interval,
            lookback=max(lookback, 30),
        )

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        a, b, c = candles[-3:]
        signals: List[Signal] = []

        if all(x.is_bullish for x in (a, b, c)) and a.close < b.close < c.close:
            if all(x.body >= x.range * 0.3 for x in (a, b, c)):
                metadata = {
                    "a_close": a.close,
                    "b_close": b.close,
                    "c_close": c.close,
                }
                signals.append(self.make_signal(symbol, "LONG", confidence=1.1, metadata=metadata))

        if all(x.is_bearish for x in (a, b, c)) and a.close > b.close > c.close:
            if all(x.body >= x.range * 0.3 for x in (a, b, c)):
                metadata = {
                    "a_close": a.close,
                    "b_close": b.close,
                    "c_close": c.close,
                }
                signals.append(self.make_signal(symbol, "SHORT", confidence=1.1, metadata=metadata))
        return signals


__all__ = ["ThreeLineStrikeStrategy"]
