"""Triple EMA squeeze breakout strategy (T3E)."""

from __future__ import annotations

import math
from statistics import median, mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import atr, ema


class TripleEMASqueezeStrategy(ModuleBase):
    """Identify breakouts when EMAs align and volatility expands."""

    def __init__(
        self,
        client,
        *,
        interval: str = "5m",
        lookback: int = 220,
        ema_fast_period: int = 9,
        ema_mid_period: int = 21,
        ema_slow_period: int = 55,
        atr_period: int = 14,
        atr_median_period: int = 50,
        breakout_period: int = 10,
        volume_window: int = 20,
    ) -> None:
        minimum_history = max(
            ema_slow_period + 2,
            atr_period + atr_median_period,
            breakout_period + 2,
            volume_window + 2,
        )
        super().__init__(
            client,
            name="Triple EMA Squeeze Breakout",
            abbreviation="T3E",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._ema_fast_period = ema_fast_period
        self._ema_mid_period = ema_mid_period
        self._ema_slow_period = ema_slow_period
        self._atr_period = atr_period
        self._atr_median_period = atr_median_period
        self._breakout_period = breakout_period
        self._volume_window = volume_window

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback:
            return []

        closes = [candle.close for candle in candles]
        ema_fast_values = ema(closes, self._ema_fast_period)
        ema_mid_values = ema(closes, self._ema_mid_period)
        ema_slow_values = ema(closes, self._ema_slow_period)
        if (
            not ema_fast_values
            or not ema_mid_values
            or not ema_slow_values
            or math.isnan(ema_fast_values[-1])
            or math.isnan(ema_mid_values[-1])
            or math.isnan(ema_slow_values[-1])
        ):
            return []

        ema_fast_current = ema_fast_values[-1]
        ema_mid_current = ema_mid_values[-1]
        ema_slow_current = ema_slow_values[-1]

        atr_values = atr(candles, self._atr_period)
        if not atr_values or math.isnan(atr_values[-1]):
            return []
        atr_slice = [
            value
            for value in atr_values[-self._atr_median_period :]
            if not math.isnan(value)
        ]
        if not atr_slice:
            return []
        median_atr = median(atr_slice)
        current_atr = atr_values[-1]
        if current_atr <= median_atr:
            return []

        volume_slice = candles[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        current_volume = candles[-1].volume
        volume_ratio = current_volume / average_volume

        recent_high = max(candle.high for candle in candles[-(self._breakout_period + 1) : -1])
        recent_low = min(candle.low for candle in candles[-(self._breakout_period + 1) : -1])
        current_close = candles[-1].close

        ema_order_metadata = {
            "ema_fast": ema_fast_current,
            "ema_mid": ema_mid_current,
            "ema_slow": ema_slow_current,
        }

        signals: List[Signal] = []

        if (
            ema_fast_current > ema_mid_current > ema_slow_current
            and current_close > recent_high
            and volume_ratio >= 1.2
        ):
            metadata = {
                "ema_order": ema_order_metadata,
                "atr_value": current_atr,
                "breakout_level": recent_high,
                "volume_ratio": volume_ratio,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "LONG",
                    confidence=1.15,
                    metadata=metadata,
                )
            )

        if (
            ema_fast_current < ema_mid_current < ema_slow_current
            and current_close < recent_low
            and volume_ratio >= 1.2
        ):
            metadata = {
                "ema_order": ema_order_metadata,
                "atr_value": current_atr,
                "breakout_level": recent_low,
                "volume_ratio": volume_ratio,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "SHORT",
                    confidence=1.15,
                    metadata=metadata,
                )
            )

        return signals


__all__ = ["TripleEMASqueezeStrategy"]
