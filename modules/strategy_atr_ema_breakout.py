"""ATR and EMA breakout strategy (BRK)."""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import atr, ema


class ATRBreakoutStrategy(ModuleBase):
    """Emit signals when price breaks key levels with strong volatility."""

    def __init__(
        self,
        client,
        *,
        interval: str = "5m",
        lookback: int = 200,
        breakout_period: int = 20,
        atr_period: int = 14,
        atr_median_period: int = 50,
        ema_fast_period: int = 20,
        ema_slow_period: int = 50,
        volume_window: int = 20,
    ) -> None:
        minimum_history = max(
            breakout_period + 1,
            atr_period + atr_median_period,
            ema_slow_period + 2,
            volume_window + 2,
        )
        super().__init__(
            client,
            name="ATR + EMA Breakout",
            abbreviation="BRK",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._breakout_period = breakout_period
        self._atr_period = atr_period
        self._atr_median_period = atr_median_period
        self._ema_fast_period = ema_fast_period
        self._ema_slow_period = ema_slow_period
        self._volume_window = volume_window

    def _identify_breakouts(
        self, candles: Sequence[Kline]
    ) -> List[Dict[str, object]]:
        series = list(candles)
        if len(series) < self.lookback:
            return []

        current = series[-1]
        window = series[-(self._breakout_period + 1) : -1]
        if not window:
            return []
        recent_high = max(candle.high for candle in window)
        recent_low = min(candle.low for candle in window)

        atr_values = atr(series, self._atr_period)
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

        closes = [candle.close for candle in series]
        ema_fast_values = ema(closes, self._ema_fast_period)
        ema_slow_values = ema(closes, self._ema_slow_period)
        if (
            not ema_fast_values
            or not ema_slow_values
            or math.isnan(ema_fast_values[-1])
            or math.isnan(ema_slow_values[-1])
        ):
            return []
        ema_fast_current = ema_fast_values[-1]
        ema_slow_current = ema_slow_values[-1]

        volume_slice = series[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        volume_ratio = current.volume / average_volume

        candidates: List[Dict[str, object]] = []

        if (
            current.close > recent_high
            and ema_fast_current > ema_slow_current
            and volume_ratio > 1.0
        ):
            breakout_distance = current.close - recent_high
            signal_strength = (
                breakout_distance / current_atr if current_atr > 0 else 0.0
            )
            metadata = {
                "atr_value": current_atr,
                "breakout_level": recent_high,
                "ema_trend": "bullish",
                "volume_ratio": volume_ratio,
                "signal_strength": signal_strength,
            }
            candidates.append({"side": "LONG", "metadata": metadata})

        if (
            current.close < recent_low
            and ema_fast_current < ema_slow_current
            and volume_ratio > 1.0
        ):
            breakout_distance = recent_low - current.close
            signal_strength = (
                breakout_distance / current_atr if current_atr > 0 else 0.0
            )
            metadata = {
                "atr_value": current_atr,
                "breakout_level": recent_low,
                "ema_trend": "bearish",
                "volume_ratio": volume_ratio,
                "signal_strength": signal_strength,
            }
            candidates.append({"side": "SHORT", "metadata": metadata})

        return candidates

    def _ema_trend(
        self, candles: Sequence[Kline]
    ) -> Optional[Tuple[str, float, float]]:
        series = list(candles)
        if len(series) < self._ema_slow_period:
            return None

        closes = [candle.close for candle in series]
        ema_fast_values = ema(closes, self._ema_fast_period)
        ema_slow_values = ema(closes, self._ema_slow_period)
        if (
            not ema_fast_values
            or not ema_slow_values
            or math.isnan(ema_fast_values[-1])
            or math.isnan(ema_slow_values[-1])
        ):
            return None

        ema_fast_current = ema_fast_values[-1]
        ema_slow_current = ema_slow_values[-1]
        if ema_fast_current > ema_slow_current:
            trend = "bullish"
        elif ema_fast_current < ema_slow_current:
            trend = "bearish"
        else:
            return None
        return trend, ema_fast_current, ema_slow_current

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback:
            return []

        candidates = self._identify_breakouts(candles)
        return [
            self.make_signal(
                symbol,
                candidate["side"],
                confidence=1.1,
                metadata=candidate["metadata"],
            )
            for candidate in candidates
        ]

    def process_with_timeframes(
        self,
        symbol: str,
        primary_candles: Sequence[Kline],
        extra_candles: Mapping[str, Sequence[Kline]],
    ) -> Iterable[Signal]:
        candidates = self._identify_breakouts(primary_candles)
        if not candidates:
            return []

        h1_candles = extra_candles.get("1h")
        if not h1_candles:
            return []

        higher_trend = self._ema_trend(h1_candles)
        if not higher_trend:
            return []

        trend_direction, ema_fast_value, ema_slow_value = higher_trend

        signals: List[Signal] = []
        for candidate in candidates:
            side = candidate["side"]
            metadata = dict(candidate["metadata"])
            if side == "LONG" and trend_direction != "bullish":
                continue
            if side == "SHORT" and trend_direction != "bearish":
                continue

            metadata.update(
                {
                    "trend_timeframe": "1h",
                    "higher_trend": trend_direction,
                    "higher_ema_fast": ema_fast_value,
                    "higher_ema_slow": ema_slow_value,
                }
            )
            signals.append(
                self.make_signal(
                    symbol,
                    side,
                    confidence=1.1,
                    metadata=metadata,
                )
            )

        return signals


__all__ = ["ATRBreakoutStrategy"]
