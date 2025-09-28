from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class BreakoutHighStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Breakout High",
            abbreviation="BKH",
            interval="1m",
            lookback=30,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 52:
            return []

        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        highs = [c.high for c in candles[:-1]]
        last = candles[-1]
        prev = candles[-2]

        reference_high = max(highs)

        median_vol = sorted(vols[:-1])[len(vols[:-1])//2]
        bodies = [abs(c.close - c.open) for c in candles[:-1]]
        median_body = sorted(bodies)[len(bodies)//2]
        last_body = abs(last.close - last.open)
        last_range = last.high - last.low
        ranges = [c.high - c.low for c in candles[:-1]]
        median_range = sorted(ranges)[len(ranges)//2]

        ret_pct = (last.close / prev.close) - 1.0
        sma_short = sum(closes[-20:]) / 20
        sma_long = sum(closes[-50:]) / 50

        if last.close < reference_high * (1.0 + 0.0008):
            return []
        if not (last.volume >= median_vol * 1.5):
            return []
        if not (last_body >= median_body * 1.5):
            return []
        if not (last_range >= median_range * 1.2):
            return []
        if not (ret_pct >= 0.001):
            return []
        if not (last.close >= sma_short >= sma_long):
            return []

        snapshot = {
            "reference_high": reference_high,
            "last_volume": last.volume,
            "median_volume": median_vol,
            "body_last": last_body,
            "median_body": median_body,
            "last_range": last_range,
            "median_range": median_range,
            "ret_pct": ret_pct,
            "sma_short": sma_short,
            "sma_long": sma_long
        }

        signal = self.make_signal(symbol, "LONG", confidence=0.8, metadata={"snapshot": snapshot})
        return [signal]

__all__ = ["BreakoutHighStrategy"]
