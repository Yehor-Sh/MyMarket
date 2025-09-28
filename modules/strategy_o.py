"""Engulfing and RSI confirmation strategy (ENG)."""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import atr, ema, rsi


class EngulfingRSIStrategy(ModuleBase):
    """Detect engulfing candles accompanied by momentum and volatility."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 160,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_average_period: int = 50,
        volume_window: int = 20,
        trend_ema_period: int = 50,
    ) -> None:
        minimum_history = max(
            rsi_period + 2,
            atr_period + atr_average_period,
            volume_window + 3,
            trend_ema_period + 2,
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
        self._trend_ema_period = trend_ema_period

    def _identify_candidates(self, candles: Sequence[Kline]) -> List[Dict[str, object]]:
        """Detect engulfing setups on the primary timeframe."""

        series = list(candles)
        if len(series) < self.lookback:
            return []

        previous = series[-2]
        current = series[-1]

        closes = [candle.close for candle in series]
        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return []
        current_rsi = rsi_values[-1]

        atr_values = atr(series, self._atr_period)
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

        volume_slice = series[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        volume_ratio = current.volume / average_volume

        candidates: List[Dict[str, object]] = []

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
            candidates.append({"side": "LONG", "metadata": metadata})

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
            candidates.append({"side": "SHORT", "metadata": metadata})

        return candidates

    def _analyse_higher_timeframe(
        self, candles: Sequence[Kline]
    ) -> Optional[Tuple[str, Optional[float]]]:
        series = list(candles)
        if len(series) < self._trend_ema_period:
            return None

        closes = [candle.close for candle in series]
        ema_values = ema(closes, self._trend_ema_period)
        if not ema_values or math.isnan(ema_values[-1]):
            return None

        trend = "bullish" if series[-1].close >= ema_values[-1] else "bearish"

        rsi_values = rsi(closes, self._rsi_period)
        rsi_value = None
        if rsi_values and not math.isnan(rsi_values[-1]):
            rsi_value = rsi_values[-1]

        return trend, rsi_value

    def _is_confirmed(self, side: str, context: Tuple[str, Optional[float]]) -> bool:
        trend, rsi_value = context
        if side == "LONG":
            if trend != "bullish":
                return False
            if rsi_value is not None and rsi_value > 70:
                return False
            return True
        if trend != "bearish":
            return False
        if rsi_value is not None and rsi_value < 30:
            return False
        return True

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback:
            return []

        candidates = self._identify_candidates(candles)
        return [
            self.make_signal(
                symbol,
                candidate["side"],
                confidence=1.05,
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
        candidates = self._identify_candidates(primary_candles)
        if not candidates:
            return []

        confirmations: List[Tuple[str, Tuple[str, Optional[float]]]] = []
        for interval in ("30m", "1h"):
            candles = extra_candles.get(interval)
            if not candles:
                continue
            context = self._analyse_higher_timeframe(candles)
            if context:
                confirmations.append((interval, context))

        if not confirmations:
            return []

        signals: List[Signal] = []
        for candidate in candidates:
            side = candidate["side"]
            metadata = dict(candidate["metadata"])
            matched_interval: Optional[str] = None
            matched_context: Optional[Tuple[str, Optional[float]]] = None
            for interval, context in confirmations:
                if self._is_confirmed(side, context):
                    matched_interval = interval
                    matched_context = context
                    break
            if not matched_interval or not matched_context:
                continue

            trend, rsi_value = matched_context
            metadata.update(
                {
                    "trend_timeframe": matched_interval,
                    "trend_direction": trend,
                }
            )
            if rsi_value is not None:
                metadata["higher_rsi"] = rsi_value

            signals.append(
                self.make_signal(
                    symbol,
                    side,
                    confidence=1.05,
                    metadata=metadata,
                )
            )

        return signals


__all__ = ["EngulfingRSIStrategy"]
