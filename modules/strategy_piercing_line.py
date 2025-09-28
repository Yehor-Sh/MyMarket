from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal


class PiercingLineStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Piercing Line",
            abbreviation="PLN",
            interval="15m",
            lookback=5,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 3:
            return []

        c1, c2 = candles[-2], candles[-1]

        if c1.close < c1.open and c2.open < c1.close and c2.close > (c1.open + c1.close) / 2:
            snapshot = {
                "c1_open": c1.open, "c1_close": c1.close,
                "c2_open": c2.open, "c2_close": c2.close
            }
            signal = self.make_signal(symbol, "LONG", confidence=0.7, metadata=snapshot)
            return [signal]

        return []


__all__ = ["PiercingLineStrategy"]
