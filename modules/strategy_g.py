from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class MovingAverageCrossStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="MA Cross Trend",
            abbreviation="MAC",
            interval="1m",
            lookback=30,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 31:
            return []
        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]
        sma_fast = sum(closes[-10:]) / 10
        sma_slow = sum(closes[-30:]) / 30
        v_med = sorted(vols[-30:])[15]
        signals = []
        if sma_fast > sma_slow and last.close >= sma_fast and last.volume >= v_med * 1.3:
            snapshot = {"pattern": "ma_cross_up", "sma_fast": sma_fast, "sma_slow": sma_slow}
            signals.append(self.make_signal(symbol, "LONG", confidence=0.7, metadata={"snapshot": snapshot}))
        elif sma_fast < sma_slow and last.close <= sma_fast and last.volume >= v_med * 1.3:
            snapshot = {"pattern": "ma_cross_down", "sma_fast": sma_fast, "sma_slow": sma_slow}
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.7, metadata={"snapshot": snapshot}))
        return signals

__all__ = ["MovingAverageCrossStrategy"]
