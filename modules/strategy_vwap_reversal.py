"""VWAP based trend reversal strategy (VWP)."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import rsi, vwap


class VWAPTrendReversalStrategy(ModuleBase):
    """Detect reversals when price crosses VWAP with momentum confirmation."""

    def __init__(
        self,
        client,
        *,
        interval: str = "5m",
        lookback: int = 200,
        rsi_period: int = 14,
        volume_window: int = 20,
    ) -> None:
        minimum_history = max(
            rsi_period + 3,
            volume_window + 3,
        )
        super().__init__(
            client,
            name="VWAP Trend Reversal",
            abbreviation="VWP",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._rsi_period = rsi_period
        self._volume_window = volume_window

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback or len(candles) < 3:
            return []

        closes = [candle.close for candle in candles]
        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]) or math.isnan(rsi_values[-2]):
            return []
        rsi_current = rsi_values[-1]

        vwap_values = vwap(candles)
        if len(vwap_values) < 2 or math.isnan(vwap_values[-1]) or math.isnan(vwap_values[-2]):
            return []
        vwap_previous = vwap_values[-2]
        vwap_current = vwap_values[-1]

        previous_close = candles[-2].close
        current = candles[-1]

        volume_slice = candles[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        volume_ratio = current.volume / average_volume

        deviation = (
            (current.close - vwap_current) / vwap_current if vwap_current else 0.0
        )

        signals: List[Signal] = []

        if (
            previous_close < vwap_previous
            and current.close > vwap_current
            and rsi_current < 35
            and volume_ratio >= 1.2
        ):
            metadata = {
                "vwap_value": vwap_current,
                "rsi_value": rsi_current,
                "volume_ratio": volume_ratio,
                "deviation_from_vwap": deviation,
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
            previous_close > vwap_previous
            and current.close < vwap_current
            and rsi_current > 65
            and volume_ratio >= 1.2
        ):
            metadata = {
                "vwap_value": vwap_current,
                "rsi_value": rsi_current,
                "volume_ratio": volume_ratio,
                "deviation_from_vwap": deviation,
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


__all__ = ["VWAPTrendReversalStrategy"]
