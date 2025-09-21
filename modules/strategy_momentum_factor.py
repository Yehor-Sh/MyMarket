"""Momentum factor strategy measuring percentage change over N bars."""

from __future__ import annotations

from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class MomentumFactorStrategy(ModuleBase):
    """Capture short term momentum bursts in the direction of the trend."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 60,
        momentum_window: int = 10,
        threshold: float = 0.01,
    ) -> None:
        minimum_history = max(lookback, momentum_window + 2)
        super().__init__(
            client,
            name="MomentumFactor",
            abbreviation="MOM",
            interval=interval,
            lookback=minimum_history,
        )
        self._momentum_window = momentum_window
        self._threshold = threshold

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) <= self._momentum_window:
            return []

        closes = [c.close for c in candles]
        current_close = closes[-1]
        reference_close = closes[-self._momentum_window]
        if reference_close == 0:
            return []

        ret = (current_close / reference_close) - 1.0
        metadata = {
            "return": ret,
            "current_close": current_close,
            "reference_close": reference_close,
        }

        if ret >= self._threshold:
            return [self.make_signal(symbol, "LONG", confidence=1.0, metadata=metadata)]

        if ret <= -self._threshold:
            return [self.make_signal(symbol, "SHORT", confidence=1.0, metadata=metadata)]

        return []


__all__ = ["MomentumFactorStrategy"]
