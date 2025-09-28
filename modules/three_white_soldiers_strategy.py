from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal


class ThreeWhiteSoldiersStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Three White Soldiers",
            abbreviation="TWS",
            interval="15m",
            lookback=6,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 4:
            return []

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        if c1.close > c1.open and c2.close > c2.open and c3.close > c3.open and c2.open > c1.open and c3.open > c2.open:
            snapshot = {"c1": (c1.open, c1.close), "c2": (c2.open, c2.close), "c3": (c3.open, c3.close)}
            signal = self.make_signal(symbol, "LONG", confidence=0.9, metadata=snapshot)
            return [signal]

        return []


__all__ = ["ThreeWhiteSoldiersStrategy"]
