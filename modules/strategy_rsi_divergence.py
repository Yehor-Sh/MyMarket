"""RSI divergence reversal strategy (DIV)."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import rsi


class RSIDivergenceStrategy(ModuleBase):
    """Look for momentum divergences near recent extremes."""

    def __init__(
        self,
        client,
        *,
        interval: str = "1h",
        lookback: int = 220,
        rsi_period: int = 14,
        divergence_window: int = 30,
        volume_window: int = 20,
    ) -> None:
        minimum_history = max(
            rsi_period + divergence_window,
            volume_window + 2,
        )
        super().__init__(
            client,
            name="RSI Divergence Reversal",
            abbreviation="DIV",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._rsi_period = rsi_period
        self._divergence_window = divergence_window
        self._volume_window = volume_window

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback or len(candles) < self._divergence_window + 2:
            return []

        current = candles[-1]
        closes = [candle.close for candle in candles]
        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return []
        current_rsi = rsi_values[-1]

        prior_candles = candles[-(self._divergence_window + 1) : -1]
        if len(prior_candles) < self._divergence_window:
            return []

        prior_rsi_values = [
            value
            for value in rsi_values[-(self._divergence_window + 1) : -1]
            if not math.isnan(value)
        ]
        if not prior_rsi_values:
            return []

        volume_slice = candles[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        volume_ratio = current.volume / average_volume

        signals: List[Signal] = []

        prior_lows = [candle.low for candle in prior_candles]
        prior_highs = [candle.high for candle in prior_candles]

        prior_min_low = min(prior_lows)
        prior_max_high = max(prior_highs)
        prior_min_rsi = min(prior_rsi_values)
        prior_max_rsi = max(prior_rsi_values)

        if (
            current.low < prior_min_low
            and current_rsi > prior_min_rsi
            and current_rsi < 30
            and volume_ratio >= 1.0
        ):
            metadata = {
                "rsi_value": current_rsi,
                "price_divergence": prior_min_low - current.low,
                "rsi_divergence": current_rsi - prior_min_rsi,
                "support_level": prior_min_low,
            }
            signals.append(
                self.make_signal(
                    symbol,
                    "LONG",
                    confidence=1.1,
                    metadata=metadata,
                )
            )

        if (
            current.high > prior_max_high
            and current_rsi < prior_max_rsi
            and current_rsi > 70
            and volume_ratio >= 1.0
        ):
            metadata = {
                "rsi_value": current_rsi,
                "price_divergence": current.high - prior_max_high,
                "rsi_divergence": prior_max_rsi - current_rsi,
                "resistance_level": prior_max_high,
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


__all__ = ["RSIDivergenceStrategy"]
