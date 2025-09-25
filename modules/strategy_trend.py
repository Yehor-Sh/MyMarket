"""Trend continuation strategy using EMA bounce confirmation."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import ema


class EMABounceStrategy(ModuleBase):
    """Look for pullbacks to EMA support/resistance within established trends."""

    def __init__(
        self,
        client,
        *,
        interval: str = "1h",
        lookback: int = 160,
    ) -> None:
        super().__init__(
            client,
            name="EMABounce",
            abbreviation="EMB",
            interval=interval,
            lookback=max(lookback, 120),
        )
        self._ema_fast = 20
        self._ema_slow = 50
        self._volume_window = 10

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < 3:
            return []

        closes = [c.close for c in candles]
        ema_fast = ema(closes, self._ema_fast)
        ema_slow = ema(closes, self._ema_slow)
        if (
            len(ema_fast) < 2
            or len(ema_slow) < 2
            or math.isnan(ema_fast[-1])
            or math.isnan(ema_fast[-2])
            or math.isnan(ema_slow[-1])
            or math.isnan(ema_slow[-2])
        ):
            return []

        if len(candles) <= self._volume_window:
            return []
        volume_slice = candles[-(self._volume_window + 1) : -1]
        avg_volume = mean(c.volume for c in volume_slice)
        if avg_volume <= 0:
            return []

        prev = candles[-2]
        curr = candles[-1]
        signals: List[Signal] = []

        # Long continuation setup.
        trend_up = ema_fast[-1] > ema_slow[-1] and ema_fast[-1] >= ema_fast[-2]
        pullback_to_ema = prev.low <= ema_fast[-2] or prev.low <= ema_slow[-2]
        resumed_up = curr.is_bullish and curr.close > ema_fast[-1]
        volume_ok = curr.volume > avg_volume
        if trend_up and pullback_to_ema and resumed_up and volume_ok:
            metadata = {
                "ema_fast": ema_fast[-1],
                "ema_slow": ema_slow[-1],
                "avg_volume": avg_volume,
                "volume": curr.volume,
                "direction": "up",
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=1.0, metadata=metadata))

        # Short continuation setup.
        trend_down = ema_fast[-1] < ema_slow[-1] and ema_fast[-1] <= ema_fast[-2]
        pullback_to_ema_short = prev.high >= ema_fast[-2] or prev.high >= ema_slow[-2]
        resumed_down = curr.is_bearish and curr.close < ema_fast[-1]
        if trend_down and pullback_to_ema_short and resumed_down and volume_ok:
            metadata = {
                "ema_fast": ema_fast[-1],
                "ema_slow": ema_slow[-1],
                "avg_volume": avg_volume,
                "volume": curr.volume,
                "direction": "down",
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=1.0, metadata=metadata))

        return signals


__all__ = ["EMABounceStrategy"]
