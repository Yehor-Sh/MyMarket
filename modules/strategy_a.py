from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class VolumeSurgeStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Volume Surge",
            abbreviation="VOL",
            interval="1m",
            lookback=50,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 21:
            return []

        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]
        prev = candles[-2]

        median_vol = sorted(vols[:-1])[len(vols[:-1])//2]
        body_last = abs(last.close - last.open)
        bodies = [abs(c.close - c.open) for c in candles[:-1]]
        median_body = sorted(bodies)[len(bodies)//2]

        last_range = last.high - last.low
        ranges = [c.high - c.low for c in candles[:-1]]
        median_range = sorted(ranges)[len(ranges)//2]

        ret_pct = (last.close / prev.close) - 1.0

        sma_val = sum(closes[-20:]) / 20

        if not (last.volume >= median_vol * 3.0):
            return []
        if not (body_last >= median_body * 2.0):
            return []
        if not (last_range >= median_range * 1.5):
            return []
        if not (abs(ret_pct) >= 0.002):
            return []

        direction = "LONG" if last.close >= sma_val else "SHORT"

        snapshot = {
            "last_volume": last.volume,
            "median_volume": median_vol,
            "body_last": body_last,
            "median_body": median_body,
            "range_last": last_range,
            "median_range": median_range,
            "ret_pct": ret_pct,
            "sma": sma_val
        }

        signal = self.make_signal(symbol, direction, confidence=0.7, metadata={"snapshot": snapshot})
        return [signal]

__all__ = ["VolumeSurgeStrategy"]
