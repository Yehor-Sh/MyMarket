from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class MomentumRocStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Momentum ROC",
            abbreviation="ROC",
            interval="1m",
            lookback=60,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 61:
            return []

        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]

        ref_price = closes[-11]
        roc = (last.close - ref_price) / ref_price if ref_price > 0 else 0

        median_vol = sorted(vols[-21:-1])[len(vols[-21:-1])//2]
        last_vol = vols[-1]

        sma_short = sum(closes[-20:]) / 20
        sma_long = sum(closes[-50:]) / 50

        if abs(roc) < 0.004:
            return []
        if not (last_vol >= median_vol * 1.3):
            return []
        if sma_short is None or sma_long is None:
            return []

        direction = None
        if roc >= 0.006 and last.close >= sma_short >= sma_long:
            direction = "LONG"
        elif roc <= -0.006 and last.close <= sma_short <= sma_long:
            direction = "SHORT"
        else:
            return []

        snapshot = {
            "roc": roc,
            "period": 10,
            "median_volume": median_vol,
            "last_vol": last_vol,
            "sma_short": sma_short,
            "sma_long": sma_long
        }

        signal = self.make_signal(symbol, direction, confidence=0.8, metadata={"snapshot": snapshot})
        return [signal]

__all__ = ["MomentumRocStrategy"]
