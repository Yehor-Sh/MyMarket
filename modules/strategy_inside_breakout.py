"""Inside bar breakout strategy with volume confirmation (INS)."""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from binance_client import Kline
from module_base import ModuleBase, Signal

from .indicators import ema


class InsideBarVolumeBreakoutStrategy(ModuleBase):
    """Breakouts following an inside bar pattern with trend confirmation."""

    def __init__(
        self,
        client,
        *,
        interval: str = "5m",
        lookback: int = 160,
        ema_fast_period: int = 20,
        ema_slow_period: int = 50,
        volume_window: int = 20,
        breakout_period: int = 10,
    ) -> None:
        minimum_history = max(
            ema_slow_period + 3,
            volume_window + 3,
            breakout_period + 3,
        )
        super().__init__(
            client,
            name="Inside Bar Breakout + Volume",
            abbreviation="INS",
            interval=interval,
            lookback=max(lookback, minimum_history),
        )
        self._ema_fast_period = ema_fast_period
        self._ema_slow_period = ema_slow_period
        self._volume_window = volume_window
        self._breakout_period = breakout_period

    def _identify_breakouts(
        self, candles: Sequence[Kline]
    ) -> List[Dict[str, object]]:
        series = list(candles)
        if len(series) < self.lookback or len(series) < 3:
            return []

        mother = series[-3]
        inside = series[-2]
        breakout = series[-1]

        if not (
            inside.high <= mother.high and inside.low >= mother.low
        ):
            return []

        closes = [candle.close for candle in series]
        ema_fast = ema(closes, self._ema_fast_period)
        ema_slow = ema(closes, self._ema_slow_period)
        if (
            not ema_fast
            or not ema_slow
            or math.isnan(ema_fast[-1])
            or math.isnan(ema_slow[-1])
        ):
            return []
        ema_fast_current = ema_fast[-1]
        ema_slow_current = ema_slow[-1]

        volume_slice = series[-(self._volume_window + 1) : -1]
        if len(volume_slice) < self._volume_window:
            return []
        average_volume = mean(candle.volume for candle in volume_slice)
        if average_volume <= 0:
            return []
        volume_ratio = breakout.volume / average_volume

        candidates: List[Dict[str, object]] = []

        recent_high = max(
            candle.high for candle in series[-(self._breakout_period + 1) : -1]
        )
        recent_low = min(
            candle.low for candle in series[-(self._breakout_period + 1) : -1]
        )

        if (
            breakout.close > mother.high
            and breakout.close > recent_high
            and ema_fast_current > ema_slow_current
            and volume_ratio >= 1.5
        ):
            metadata = {
                "mother_high": mother.high,
                "mother_low": mother.low,
                "ema_trend": "bullish",
                "volume_ratio": volume_ratio,
            }
            candidates.append({"side": "LONG", "metadata": metadata})

        if (
            breakout.close < mother.low
            and breakout.close < recent_low
            and ema_fast_current < ema_slow_current
            and volume_ratio >= 1.5
        ):
            metadata = {
                "mother_high": mother.high,
                "mother_low": mother.low,
                "ema_trend": "bearish",
                "volume_ratio": volume_ratio,
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
        ema_fast = ema(closes, self._ema_fast_period)
        ema_slow = ema(closes, self._ema_slow_period)
        if (
            not ema_fast
            or not ema_slow
            or math.isnan(ema_fast[-1])
            or math.isnan(ema_slow[-1])
        ):
            return None

        ema_fast_current = ema_fast[-1]
        ema_slow_current = ema_slow[-1]
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

        m30_candles = extra_candles.get("30m")
        if not m30_candles:
            return []

        higher_trend = self._ema_trend(m30_candles)
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
                    "trend_timeframe": "30m",
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


__all__ = ["InsideBarVolumeBreakoutStrategy"]
