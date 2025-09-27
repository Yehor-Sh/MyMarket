"""Pin bar reversal strategy with support/resistance confirmation."""

from __future__ import annotations

import math
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import ema, rsi


class PinBarStrategy(ModuleBase):
    """Identify pin bar candles near recent support or resistance."""

    def __init__(
        self,
        client,
        *,
        interval: str = "1h",
        lookback: int = 160,
        level_window: int = 50,
        tail_ratio: float = 2.0,
    ) -> None:
        super().__init__(
            client,
            name="PinBar",
            abbreviation="PIN",
            interval=interval,
            lookback=max(lookback, level_window + 60),
        )
        self._level_window = max(3, level_window)
        self._tail_ratio = tail_ratio
        self._tolerance = 0.0015
        self._ema_fast_period = 20
        self._ema_slow_period = 50
        self._rsi_period = 14

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self._level_window + 2:
            return []

        curr = candles[-1]
        recent_levels = candles[-(self._level_window + 1) : -1]

        support = min(c.low for c in recent_levels)
        resistance = max(c.high for c in recent_levels)

        signals: List[Signal] = []

        body = max(curr.body, 1e-8)
        upper_tail = curr.upper_wick
        lower_tail = curr.lower_wick

        def near(value: float, target: float) -> bool:
            if target == 0:
                return False
            return abs(value - target) <= target * self._tolerance

        closes = [c.close for c in candles]
        ema_fast_values = ema(closes, self._ema_fast_period)
        ema_slow_values = ema(closes, self._ema_slow_period)
        if (
            len(ema_fast_values) < 2
            or len(ema_slow_values) < 2
            or math.isnan(ema_fast_values[-1])
            or math.isnan(ema_slow_values[-1])
            or math.isnan(ema_fast_values[-2])
            or math.isnan(ema_slow_values[-2])
        ):
            return []

        fast_prev, fast_curr = ema_fast_values[-2], ema_fast_values[-1]
        slow_prev, slow_curr = ema_slow_values[-2], ema_slow_values[-1]

        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return []
        current_rsi = rsi_values[-1]

        cross_up = fast_prev <= slow_prev and fast_curr > slow_curr
        cross_down = fast_prev >= slow_prev and fast_curr < slow_curr

        if (
            curr.is_bullish
            and lower_tail >= body * self._tail_ratio
            and upper_tail <= body
            and (curr.low <= support or near(curr.low, support))
            and cross_up
            and current_rsi < 30.0
        ):
            metadata = {
                "support": support,
                "body": body,
                "lower_wick": lower_tail,
                "ema_fast": fast_curr,
                "ema_slow": slow_curr,
                "rsi": current_rsi,
                "ema_fast_prev": fast_prev,
                "ema_slow_prev": slow_prev,
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=0.9, metadata=metadata))

        if (
            curr.is_bearish
            and upper_tail >= body * self._tail_ratio
            and lower_tail <= body
            and (curr.high >= resistance or near(curr.high, resistance))
            and cross_down
            and current_rsi > 70.0
        ):
            metadata = {
                "resistance": resistance,
                "body": body,
                "upper_wick": upper_tail,
                "ema_fast": fast_curr,
                "ema_slow": slow_curr,
                "rsi": current_rsi,
                "ema_fast_prev": fast_prev,
                "ema_slow_prev": slow_prev,
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.9, metadata=metadata))

        return signals


__all__ = ["PinBarStrategy"]
