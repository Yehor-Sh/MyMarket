from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal
import math

class VwapReversionStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="VWAP Reversion",
            abbreviation="VWR",
            interval="1m",
            lookback=60,
        )

    def vwap(self, prices: list[float], volumes: list[float], n: int) -> float | None:
        if len(prices) < n or len(volumes) < n:
            return None
        p = prices[-n:]
        v = volumes[-n:]
        tv = sum(p[i]*v[i] for i in range(n))
        vv = sum(v)
        return tv/vv if vv > 0 else None

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 61:
            return []
        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]
        val = self.vwap(closes, vols, 60)
        if val is None:
            return []
        devs = [abs(closes[i] - val) / val for i in range(-60,0)]
        mean = sum(devs)/len(devs)
        variance = sum((x-mean)**2 for x in devs)/len(devs)
        sd = math.sqrt(variance)
        dist = (last.close - val) / val
        signals = []
        if dist >= 1.5*sd:
            snapshot = {"pattern": "vwap_reversion_short", "vwap": val, "z": dist/sd}
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.8, metadata={"snapshot": snapshot}))
        elif dist <= -1.5*sd:
            snapshot = {"pattern": "vwap_reversion_long", "vwap": val, "z": dist/sd}
            signals.append(self.make_signal(symbol, "LONG", confidence=0.8, metadata={"snapshot": snapshot}))
        return signals

__all__ = ["VwapReversionStrategy"]
