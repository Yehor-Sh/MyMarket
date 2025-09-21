"""Pin bar reversal strategy with support/resistance confirmation."""

from __future__ import annotations

from statistics import mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class PinBarStrategy(ModuleBase):
    """Identify pin bar candles near recent support or resistance."""

    def __init__(
        self,
        client,
        *,
        interval: str = "1h",
        lookback: int = 80,
        level_window: int = 8,
        tail_ratio: float = 2.0,
    ) -> None:
        super().__init__(
            client,
            name="PinBar",
            abbreviation="PIN",
            interval=interval,
            lookback=max(lookback, 60),
        )
        self._volume_window = 10
        self._level_window = max(3, level_window)
        self._tail_ratio = tail_ratio
        self._tolerance = 0.0015

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self._level_window + 2:
            return []

        curr = candles[-1]
        recent_levels = candles[-(self._level_window + 1) : -1]

        support = min(c.low for c in recent_levels)
        resistance = max(c.high for c in recent_levels)

        if len(candles) <= self._volume_window:
            return []
        volume_slice = candles[-(self._volume_window + 1) : -1]
        avg_volume = mean(c.volume for c in volume_slice)
        if avg_volume <= 0:
            return []

        signals: List[Signal] = []

        body = max(curr.body, 1e-8)
        upper_tail = curr.upper_wick
        lower_tail = curr.lower_wick
        volume_ok = curr.volume > avg_volume

        def near(value: float, target: float) -> bool:
            if target == 0:
                return False
            return abs(value - target) <= target * self._tolerance

        if (
            curr.is_bullish
            and lower_tail >= body * self._tail_ratio
            and upper_tail <= body
            and (curr.low <= support or near(curr.low, support))
            and volume_ok
        ):
            metadata = {
                "support": support,
                "body": body,
                "lower_wick": lower_tail,
                "volume": curr.volume,
                "avg_volume": avg_volume,
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=0.9, metadata=metadata))

        if (
            curr.is_bearish
            and upper_tail >= body * self._tail_ratio
            and lower_tail <= body
            and (curr.high >= resistance or near(curr.high, resistance))
            and volume_ok
        ):
            metadata = {
                "resistance": resistance,
                "body": body,
                "upper_wick": upper_tail,
                "volume": curr.volume,
                "avg_volume": avg_volume,
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.9, metadata=metadata))

        return signals


__all__ = ["PinBarStrategy"]
