"""Bollinger band squeeze breakout strategy (BBS)."""

from __future__ import annotations

import math
from statistics import median, mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import atr, bollinger_bands


class BollingerSqueezeBreakoutStrategy(ModuleBase):
    """Trade breakouts from volatility contractions with volume confirmation."""

    def __init__(
        self,
        client,
        *,
        interval: str = "1h",
        lookback: int = 220,
        bollinger_period: int = 20,
        atr_period: int = 14,
        atr_growth_bars: int = 3,
        median_window: int = 50,
        volume_window: int = 20,
        stddev_multiplier: float = 2.0,
    ) -> None:
        minimum_history = max(
            bollinger_period + median_window,
            atr_period + atr_growth_bars,
            volume_window + 2,
        )
        super().__init__(
            client,
            name="Bollinger Band Squeeze Breakout",
            abbreviation="BBS",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._bollinger_period = bollinger_period
        self._atr_period = atr_period
        self._atr_growth_bars = atr_growth_bars
        self._median_window = median_window
        self._volume_window = volume_window
        self._stddev_multiplier = stddev_multiplier

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback:
            return []

        closes = [candle.close for candle in candles]
        _middle, upper, lower, width = bollinger_bands(
            closes,
            self._bollinger_period,
            stddev_multiplier=self._stddev_multiplier,
        )
        if not width or math.isnan(width[-1]):
            return []
        current_width = width[-1]
        width_slice = [
            value
            for value in width[-self._median_window :]
            if not math.isnan(value)
        ]
        if not width_slice:
            return []
        median_width = median(width_slice)
        if median_width <= 0 or current_width >= 0.7 * median_width:
            return []

        atr_values = atr(candles, self._atr_period)
        if len(atr_values) < self._atr_growth_bars + 1:
            return []
        recent_atr = atr_values[-self._atr_growth_bars :]
        if any(math.isnan(value) for value in recent_atr):
            return []
        atr_increasing = all(
            recent_atr[idx] > recent_atr[idx - 1]
            for idx in range(1, len(recent_atr))
        )
        if not atr_increasing:
            return []
        current_atr = recent_atr[-1]

        volume_slice = candles[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        current = candles[-1]
        volume_ratio = current.volume / average_volume
        if volume_ratio < 1.3:
            return []

        upper_band = upper[-1]
        lower_band = lower[-1]
        if math.isnan(upper_band) or math.isnan(lower_band):
            return []

        signals: List[Signal] = []

        if current.close > upper_band:
            breakout_strength = (
                (current.close - upper_band) / upper_band if upper_band else 0.0
            )
            metadata = {
                "band_width": current_width,
                "atr_value": current_atr,
                "breakout_strength": breakout_strength,
                "volume_ratio": volume_ratio,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "LONG",
                    confidence=1.1,
                    metadata=metadata,
                )
            )

        if current.close < lower_band:
            breakout_strength = (
                (lower_band - current.close) / lower_band if lower_band else 0.0
            )
            metadata = {
                "band_width": current_width,
                "atr_value": current_atr,
                "breakout_strength": breakout_strength,
                "volume_ratio": volume_ratio,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "SHORT",
                    confidence=1.1,
                    metadata=metadata,
                )
            )

        return signals


__all__ = ["BollingerSqueezeBreakoutStrategy"]
