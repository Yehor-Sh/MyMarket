"""Mean reversion strategy using z-score of the last close."""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class MeanReversionStrategy(ModuleBase):
    """Fade extremes when price deviates strongly from its moving average."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 120,
        window: int = 20,
        z_threshold: float = 2.0,
    ) -> None:
        minimum_history = max(lookback, window + 2)
        super().__init__(
            client,
            name="MeanReversion",
            abbreviation="MRV",
            interval=interval,
            lookback=minimum_history,
        )
        self._window = window
        self._z_threshold = z_threshold

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) <= self._window:
            return []

        closes = [candle.close for candle in candles]
        lookback_slice = closes[-self._window :]
        avg_close = fmean(lookback_slice)
        deviation = pstdev(lookback_slice)
        if deviation == 0:
            return []

        last_close = closes[-1]
        z_score = (last_close - avg_close) / deviation
        metadata = {
            "mean": avg_close,
            "std_dev": deviation,
            "z_score": z_score,
            "last_close": last_close,
        }

        if z_score >= self._z_threshold:
            return [self.make_signal(symbol, "SHORT", confidence=0.9, metadata=metadata)]

        if z_score <= -self._z_threshold:
            return [self.make_signal(symbol, "LONG", confidence=0.9, metadata=metadata)]

        return []


__all__ = ["MeanReversionStrategy"]
