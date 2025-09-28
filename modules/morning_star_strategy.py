from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal


class MorningStarStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Morning Star",
            abbreviation="MSTAR",
            interval="15m",
            lookback=5,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 4:
            return []

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        if c1.close < c1.open and abs(c2.close - c2.open) < (c1.open - c1.close) * 0.5 and c3.close > (c1.open + c1.close) / 2:
            snapshot = {"c1": (c1.open, c1.close), "c2": (c2.open, c2.close), "c3": (c3.open, c3.close)}
            signal = self.make_signal(symbol, "LONG", confidence=0.85, metadata=snapshot)
            return [signal]

        return []


__all__ = ["MorningStarStrategy"]
