"""RSI divergence reversal strategy (DIV)."""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import rsi


class RSIDivergenceStrategy(ModuleBase):
    """Look for momentum divergences near recent extremes."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
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

    def _analyse_candles(self, candles: Sequence[Kline]) -> Dict[str, Dict[str, float]]:
        """Compute divergence context for ``candles``."""

        result: Dict[str, Dict[str, float]] = {}
        series = list(candles)
        if len(series) < self._divergence_window + 2:
            return result

        current = series[-1]
        closes = [candle.close for candle in series]
        rsi_values = rsi(closes, self._rsi_period)
        if not rsi_values or math.isnan(rsi_values[-1]):
            return result
        current_rsi = rsi_values[-1]

        prior_slice = series[-(self._divergence_window + 1) : -1]
        if len(prior_slice) < self._divergence_window:
            return result

        prior_rsi_values = [
            value
            for value in rsi_values[-(self._divergence_window + 1) : -1]
            if not math.isnan(value)
        ]
        if not prior_rsi_values:
            return result

        prior_lows = [candle.low for candle in prior_slice]
        prior_highs = [candle.high for candle in prior_slice]
        prior_min_low = min(prior_lows)
        prior_max_high = max(prior_highs)
        prior_min_rsi = min(prior_rsi_values)
        prior_max_rsi = max(prior_rsi_values)

        volume_ratio: Optional[float] = None
        volume_slice = series[-(self._volume_window + 1) : -1]
        if len(volume_slice) >= self._volume_window:
            average_volume = mean(candle.volume for candle in volume_slice)
            if average_volume > 0:
                volume_ratio = current.volume / average_volume

        result["LONG"] = {
            "divergence": current.low < prior_min_low and current_rsi > prior_min_rsi,
            "rsi": current_rsi,
            "support_level": prior_min_low,
            "price_divergence": prior_min_low - current.low,
            "rsi_divergence": current_rsi - prior_min_rsi,
            "volume_ratio": volume_ratio,
        }
        result["SHORT"] = {
            "divergence": current.high > prior_max_high and current_rsi < prior_max_rsi,
            "rsi": current_rsi,
            "resistance_level": prior_max_high,
            "price_divergence": current.high - prior_max_high,
            "rsi_divergence": prior_max_rsi - current_rsi,
            "volume_ratio": volume_ratio,
        }
        return result

    def _primary_ready(self, side: str, context: Dict[str, float]) -> bool:
        volume_ratio = context.get("volume_ratio") or 0.0
        if side == "LONG":
            support = float(context.get("support_level") or 0.0)
            rsi_value = context.get("rsi")
            return bool(
                context.get("divergence")
                and rsi_value is not None
                and rsi_value < 30
                and volume_ratio >= 1.0
                and support > 0
            )
        resistance = float(context.get("resistance_level") or 0.0)
        rsi_value = context.get("rsi")
        return bool(
            context.get("divergence")
            and rsi_value is not None
            and rsi_value > 70
            and volume_ratio >= 1.0
            and resistance > 0
        )

    def _build_metadata(self, side: str, context: Dict[str, float]) -> Dict[str, float]:
        if side == "LONG":
            return {
                "rsi_value": context["rsi"],
                "price_divergence": context["price_divergence"],
                "rsi_divergence": context["rsi_divergence"],
                "support_level": context["support_level"],
            }
        return {
            "rsi_value": context["rsi"],
            "price_divergence": context["price_divergence"],
            "rsi_divergence": context["rsi_divergence"],
            "resistance_level": context["resistance_level"],
        }

    def _confirm_long(self, context: Optional[Dict[str, float]]) -> bool:
        if not context or not context.get("divergence"):
            return False
        rsi_value = context.get("rsi")
        support = context.get("support_level")
        return bool(rsi_value is not None and rsi_value <= 30 and support and support > 0)

    def _confirm_short(self, context: Optional[Dict[str, float]]) -> bool:
        if not context or not context.get("divergence"):
            return False
        rsi_value = context.get("rsi")
        resistance = context.get("resistance_level")
        return bool(rsi_value is not None and rsi_value >= 70 and resistance and resistance > 0)

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self.lookback or len(candles) < self._divergence_window + 2:
            return []

        analysis = self._analyse_candles(candles)
        if not analysis:
            return []

        signals: List[Signal] = []
        long_context = analysis.get("LONG")
        if long_context and self._primary_ready("LONG", long_context):
            metadata = self._build_metadata("LONG", long_context)
            signals.append(
                self.make_signal(
                    symbol,
                    "LONG",
                    confidence=1.1,
                    metadata=metadata,
                )
            )

        short_context = analysis.get("SHORT")
        if short_context and self._primary_ready("SHORT", short_context):
            metadata = self._build_metadata("SHORT", short_context)
            signals.append(
                self.make_signal(
                    symbol,
                    "SHORT",
                    confidence=1.1,
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
        analysis = self._analyse_candles(primary_candles)
        if not analysis:
            return []

        h1_candles = extra_candles.get("1h")
        if not h1_candles:
            return []

        higher_analysis = self._analyse_candles(h1_candles)
        if not higher_analysis:
            return []

        signals: List[Signal] = []

        long_context = analysis.get("LONG")
        higher_long = higher_analysis.get("LONG")
        if (
            long_context
            and self._primary_ready("LONG", long_context)
            and self._confirm_long(higher_long)
        ):
            metadata = self._build_metadata("LONG", long_context)
            metadata.update(
                {
                    "confirmation_timeframe": "1h",
                    "h1_rsi": higher_long["rsi"],
                    "h1_support_level": higher_long["support_level"],
                }
            )
            signals.append(
                self.make_signal(
                    symbol,
                    "LONG",
                    confidence=1.1,
                    metadata=metadata,
                )
            )

        short_context = analysis.get("SHORT")
        higher_short = higher_analysis.get("SHORT")
        if (
            short_context
            and self._primary_ready("SHORT", short_context)
            and self._confirm_short(higher_short)
        ):
            metadata = self._build_metadata("SHORT", short_context)
            metadata.update(
                {
                    "confirmation_timeframe": "1h",
                    "h1_rsi": higher_short["rsi"],
                    "h1_resistance_level": higher_short["resistance_level"],
                }
            )
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
