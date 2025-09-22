"""Engulfing and RSI confirmation strategy (ENG)."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import atr, rsi


class EngulfingRSIStrategy(ModuleBase):
    """Detect engulfing candles accompanied by momentum and volatility."""

    def __init__(
        self,
        client,
        *,
        interval: str = "1h",
        lookback: int = 160,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_average_period: int = 50,
        volume_window: int = 20,
    ) -> None:
        minimum_history = max(
            rsi_period + 2,
            atr_period + atr_average_period,
            volume_window + 3,
        )
        super().__init__(
            client,
            name="Engulfing + RSI",
            abbreviation="ENG",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._rsi_period = rsi_period
        self._atr_period = atr_period
        self._atr_average_period = atr_average_period
        self._volume_window = volume_window

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback:
            return []

        previous = candles[-2]
        current = candles[-1]

        closes = [candle.close for candle in candles]
        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return []
        current_rsi = rsi_values[-1]

        atr_values = atr(candles, self._atr_period)
        if not atr_values or math.isnan(atr_values[-1]):
            return []
        atr_slice = [
            value
            for value in atr_values[-self._atr_average_period :]
            if not math.isnan(value)
        ]
        if not atr_slice:
            return []
        atr_average = sum(atr_slice) / len(atr_slice)
        current_atr = atr_values[-1]
        if current_atr <= atr_average:
            return []

        volume_slice = candles[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        volume_ratio = current.volume / average_volume

        signals: List[Signal] = []

        bullish_engulf = (
            previous.is_bearish
            and current.is_bullish
            and current.open <= previous.close
            and current.close >= previous.open
        )
        bearish_engulf = (
            previous.is_bullish
            and current.is_bearish
            and current.open >= previous.close
            and current.close <= previous.open
        )

        if (
            bullish_engulf
            and current_rsi < 25
            and volume_ratio >= 1.2
        ):
            body_ratio = current.body / max(previous.body, 1e-9)
            metadata = {
                "engulfing_size": body_ratio,
                "rsi_value": current_rsi,
                "atr_value": current_atr,
                "volume_ratio": volume_ratio,
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
            bearish_engulf
            and current_rsi > 75
            and volume_ratio >= 1.2
        ):
            body_ratio = current.body / max(previous.body, 1e-9)
            metadata = {
                "engulfing_size": body_ratio,
                "rsi_value": current_rsi,
                "atr_value": current_atr,
                "volume_ratio": volume_ratio,
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


__all__ = ["EngulfingRSIStrategy"]
