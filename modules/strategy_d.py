from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class RangeSurgeStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Range Surge",
            abbreviation="RGS",
            interval="1m",
            lookback=40,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 52:
            return []

        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]
        prev = candles[-2]

        window = candles[-41:-1]
        ranges = [c.high - c.low for c in window]
        median_range = sorted(ranges)[len(ranges)//2]

        vols_window = [c.volume for c in window]
        median_vol = sorted(vols_window)[len(vols_window)//2]

        bodies = [abs(c.close - c.open) for c in window]
        median_body = sorted(bodies)[len(bodies)//2]
        last_body = abs(last.close - last.open)

        last_range = last.high - last.low
        ret_pct = (last.close / prev.close) - 1.0

        sma_short = sum(closes[-20:]) / 20
        sma_long = sum(closes[-50:]) / 50

        if not (last_range >= median_range * 2.5):
            return []
        if not (last.volume >= median_vol * 1.5):
            return []
        if not (last_body >= median_body * 1.5):
            return []
        if not (abs(ret_pct) >= 0.001):
            return []

        direction = "LONG" if last.close > last.open else "SHORT"

        snapshot = {
            "last_range": last_range,
            "median_range": median_range,
            "last_volume": last.volume,
            "median_volume": median_vol,
            "body_last": last_body,
            "median_body": median_body,
            "ret_pct": ret_pct,
            "sma_short": sma_short,
            "sma_long": sma_long
        }

        signal = self.make_signal(symbol, direction, confidence=0.75, metadata={"snapshot": snapshot})
        return [signal]

__all__ = ["RangeSurgeStrategy"]
