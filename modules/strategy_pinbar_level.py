"""Pin bar strategy with key level and EMA confirmation (PIN)."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import ema, rsi


class PinBarLevelStrategy(ModuleBase):
    """Identify pin bars forming near key levels with momentum filters."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
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

    def _identify_pin(self, candles: Sequence[Kline]) -> Dict[str, Dict[str, float]]:
        """Return pin-bar candidates for ``candles``."""

        result: Dict[str, Dict[str, float]] = {}
        series = list(candles)
        if len(series) < self.lookback:
            return result

        pin = series[-1]
        closes = [candle.close for candle in series]

        ema_values = ema(closes, self._ema_period)
        if not ema_values or math.isnan(ema_values[-1]):
            return result
        ema_current = ema_values[-1]

        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return result
        rsi_current = rsi_values[-1]

        body = pin.body if pin.body > 0 else 1e-9

        if (
            pin.lower_wick >= 2 * body
            and pin.close > ema_current
            and rsi_current < 30
        ):
            result["LONG"] = {
                "pin_low": pin.low,
                "pin_high": pin.high,
                "pin_ratio": pin.lower_wick / body,
                "rsi_value": rsi_current,
                "ema_trend": "bullish",
            }

        if (
            pin.upper_wick >= 2 * body
            and pin.close < ema_current
            and rsi_current > 70
        ):
            result["SHORT"] = {
                "pin_low": pin.low,
                "pin_high": pin.high,
                "pin_ratio": pin.upper_wick / body,
                "rsi_value": rsi_current,
                "ema_trend": "bearish",
            }

        return result

    def _compute_levels(
        self, candles: Sequence[Kline]
    ) -> tuple[Optional[float], Optional[float]]:
        series = list(candles)
        if len(series) < self._level_window:
            return None, None
        window = series[-self._level_window :]
        support_level = min(candle.low for candle in window)
        resistance_level = max(candle.high for candle in window)
        return support_level, resistance_level

    def _build_metadata(
        self, side: str, context: Dict[str, float], level: float
    ) -> Dict[str, float]:
        metadata = {
            "pin_ratio": context["pin_ratio"],
            "rsi_value": context["rsi_value"],
            "ema_trend": context["ema_trend"],
        }
        if side == "LONG":
            metadata["support_level"] = level
        else:
            metadata["resistance_level"] = level
        return metadata

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback:
            return []

        candidates = self._identify_pin(candles)
        if not candidates:
            return []

        support_level, resistance_level = self._compute_levels(candles)

        signals: List[Signal] = []

        long_context = candidates.get("LONG")
        if long_context and support_level and support_level > 0:
            tolerance = support_level * self._level_tolerance
            if abs(long_context["pin_low"] - support_level) <= tolerance:
                metadata = self._build_metadata("LONG", long_context, support_level)
                signals.append(
                    self.make_signal(
                        symbol,
                        "LONG",
                        confidence=1.05,
                        metadata=metadata,
                    )
                )

        short_context = candidates.get("SHORT")
        if short_context and resistance_level and resistance_level > 0:
            tolerance = resistance_level * self._level_tolerance
            if abs(short_context["pin_high"] - resistance_level) <= tolerance:
                metadata = self._build_metadata("SHORT", short_context, resistance_level)
                signals.append(
                    self.make_signal(
                        symbol,
                        "SHORT",
                        confidence=1.05,
                        metadata=metadata,
                    )
                )

        return signals

    def process_with_timeframes(
        self,
        symbol: str,
        primary_candles: Sequence[Kline],
        extra_candles: Mapping[str, Sequence[Kline]],
    ) -> Iterable[Signal]:
        candidates = self._identify_pin(primary_candles)
        if not candidates:
            return []

        h1_candles = extra_candles.get("1h")
        if not h1_candles:
            return []

        support_level, resistance_level = self._compute_levels(h1_candles)
        signals: List[Signal] = []

        long_context = candidates.get("LONG")
        if long_context and support_level and support_level > 0:
            tolerance = support_level * self._level_tolerance
            if abs(long_context["pin_low"] - support_level) <= tolerance:
                metadata = self._build_metadata("LONG", long_context, support_level)
                metadata["level_timeframe"] = "1h"
                signals.append(
                    self.make_signal(
                        symbol,
                        "LONG",
                        confidence=1.05,
                        metadata=metadata,
                    )
                )

        short_context = candidates.get("SHORT")
        if short_context and resistance_level and resistance_level > 0:
            tolerance = resistance_level * self._level_tolerance
            if abs(short_context["pin_high"] - resistance_level) <= tolerance:
                metadata = self._build_metadata("SHORT", short_context, resistance_level)
                metadata["level_timeframe"] = "1h"
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
