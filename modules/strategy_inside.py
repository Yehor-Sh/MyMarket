"""Inside bar strategy."""

from __future__ import annotations

from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class InsideBarStrategy(ModuleBase):
    """Signal when the last candle is fully inside the previous range."""

    def __init__(self, client, *, interval: str = "1h", lookback: int = 60) -> None:
        super().__init__(
            client,
            name="InsideBar",
            abbreviation="INS",
            interval=interval,
            lookback=max(lookback, 25),
        )

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        prev = candles[-2]
        curr = candles[-1]
        signals: List[Signal] = []

        if curr.high <= prev.high and curr.low >= prev.low:
            direction = "LONG" if curr.close >= curr.open else "SHORT"
            metadata = {
                "prev_high": prev.high,
                "prev_low": prev.low,
                "curr_high": curr.high,
                "curr_low": curr.low,
            }
            signals.append(self.make_signal(symbol, direction, confidence=0.7, metadata=metadata))
        return signals


__all__ = ["InsideBarStrategy"]
