"""Range breakout strategy focused on intraday levels."""

from __future__ import annotations

from statistics import fmean
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class RangeBreakoutStrategy(ModuleBase):
    """Identify breakouts from a recent trading range on 15 minute bars."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 60,
        breakout_window: int = 20,
        volume_window: int = 20,
    ) -> None:
        minimum_history = max(breakout_window + 2, volume_window + 2, lookback)
        super().__init__(
            client,
            name="RangeBreakout",
            abbreviation="RBO",
            interval=interval,
            lookback=minimum_history,
        )
        self._breakout_window = breakout_window
        self._volume_window = volume_window

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        if len(candles) < self._breakout_window + 1:
            return []

        breakout_candle = candles[-1]
        prev_candle = candles[-2]
        window = candles[-(self._breakout_window + 1) : -1]
        resistance = max(candle.high for candle in window)
        support = min(candle.low for candle in window)

        avg_volume = None
        if len(candles) > self._volume_window:
            volume_slice = candles[-(self._volume_window + 1) : -1]
            avg_volume = fmean(c.volume for c in volume_slice)

        signals: List[Signal] = []

        if breakout_candle.close > resistance and prev_candle.close <= resistance:
            metadata = {
                "breakout_level": resistance,
                "previous_close": prev_candle.close,
                "current_close": breakout_candle.close,
            }
            if avg_volume is not None:
                metadata.update(
                    {
                        "volume": breakout_candle.volume,
                        "avg_volume": avg_volume,
                    }
                )
            signals.append(self.make_signal(symbol, "LONG", confidence=1.05, metadata=metadata))

        if breakout_candle.close < support and prev_candle.close >= support:
            metadata = {
                "breakdown_level": support,
                "previous_close": prev_candle.close,
                "current_close": breakout_candle.close,
            }
            if avg_volume is not None:
                metadata.update(
                    {
                        "volume": breakout_candle.volume,
                        "avg_volume": avg_volume,
                    }
                )
            signals.append(self.make_signal(symbol, "SHORT", confidence=1.05, metadata=metadata))

        return signals


__all__ = ["RangeBreakoutStrategy"]
