from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal


class DarkCloudCoverStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Dark Cloud Cover",
            abbreviation="DCC",
            interval="15m",
            lookback=5,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 3:
            return []

        c1, c2 = candles[-2], candles[-1]

        if c1.close > c1.open and c2.open > c1.high and c2.close < (c1.open + c1.close) / 2:
            snapshot = {"c1": (c1.open, c1.close), "c2": (c2.open, c2.close)}
            signal = self.make_signal(symbol, "SHORT", confidence=0.8, metadata=snapshot)
            return [signal]

        return []


__all__ = ["DarkCloudCoverStrategy"]
