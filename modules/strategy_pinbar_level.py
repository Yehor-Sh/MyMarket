"""Pin bar strategy with key level and EMA confirmation (PIN)."""

from __future__ import annotations

import math
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import ema, rsi


class PinBarLevelStrategy(ModuleBase):
    """Identify pin bars forming near key levels with momentum filters."""

    def __init__(
        self,
        client,
        *,
        interval: str = "1h",
        lookback: int = 200,
        ema_period: int = 20,
        rsi_period: int = 14,
        level_window: int = 50,
        level_tolerance: float = 0.01,
    ) -> None:
        minimum_history = max(
            ema_period + 2,
            rsi_period + 2,
            level_window + 2,
        )
        super().__init__(
            client,
            name="Pin Bar + Level + EMA",
            abbreviation="PIN",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._ema_period = ema_period
        self._rsi_period = rsi_period
        self._level_window = level_window
        self._level_tolerance = level_tolerance

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback:
            return []

        pin = candles[-1]
        closes = [candle.close for candle in candles]

        ema_values = ema(closes, self._ema_period)
        if not ema_values or math.isnan(ema_values[-1]):
            return []
        ema_current = ema_values[-1]

        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return []
        rsi_current = rsi_values[-1]

        level_slice = candles[-self._level_window :]
        if len(level_slice) < self._level_window:
            return []

        support_level = min(candle.low for candle in level_slice)
        resistance_level = max(candle.high for candle in level_slice)
        tolerance_value_support = support_level * self._level_tolerance if support_level else 0.0
        tolerance_value_resistance = (
            resistance_level * self._level_tolerance if resistance_level else 0.0
        )

        signals: List[Signal] = []
        body = pin.body
        body = body if body > 0 else 1e-9

        if (
            pin.lower_wick >= 2 * body
            and pin.close > ema_current
            and rsi_current < 30
            and support_level > 0
            and abs(pin.low - support_level) <= tolerance_value_support
        ):
            metadata = {
                "pin_ratio": pin.lower_wick / body,
                "rsi_value": rsi_current,
                "ema_trend": "bullish",
                "support_level": support_level,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "LONG",
                    confidence=1.05,
                    metadata=metadata,
                )
            )

        if (
            pin.upper_wick >= 2 * body
            and pin.close < ema_current
            and rsi_current > 70
            and resistance_level > 0
            and abs(pin.high - resistance_level) <= tolerance_value_resistance
        ):
            metadata = {
                "pin_ratio": pin.upper_wick / body,
                "rsi_value": rsi_current,
                "ema_trend": "bearish",
                "resistance_level": resistance_level,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "SHORT",
                    confidence=1.05,
                    metadata=metadata,
                )
            )

        return signals


__all__ = ["PinBarLevelStrategy"]
