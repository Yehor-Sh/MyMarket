"""ATR based volatility breakout strategy with adaptive thresholds."""

from __future__ import annotations

from typing import Iterable, Sequence

from binance_client import Kline
from module_base import ModuleBase, Signal

from .strategy_shared import base_metadata, passes_sanity


class VolatilityBreakoutStrategy(ModuleBase):
    """Expansion beyond recent range by ATR multiple, adaptive to current ATR% and volume."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 240,
        atr_period: int = 15,
        base_atr_multiplier: float = 0.9,
    ) -> None:
        minimum_history = max(lookback, atr_period + 210, 220)
        super().__init__(
            client,
            name="VolatilityBreakout",
            abbreviation="VBO",
            interval=interval,
            lookback=minimum_history,
        )
        self._atr_period = atr_period
        self._base_atr_multiplier = base_atr_multiplier

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        bars = list(candles)
        if len(bars) <= self._atr_period:
            return []

        meta = base_metadata(bars, atr_period=self._atr_period)
        if not passes_sanity(meta, min_atr_pct=0.0009, min_rel_vol=0.9):
            return []

        last = bars[-1]
        prev = bars[-2]

        atr_value = meta["atr"]
        if atr_value <= 0:
            return []

        dyn_multiplier = self._base_atr_multiplier * (1.0 + min(0.5, max(0.0, meta["atr_pct"] * 80)))
        threshold = atr_value * dyn_multiplier
        delta = last.close - prev.close

        signals: list[Signal] = []
        if delta > threshold and meta["trend"] != "DOWN" and meta["rel_volume"] >= 1.05:
            strength = (delta / atr_value) * meta["rel_volume"]
            meta_long = dict(meta)
            meta_long.update({
                "signal_strength": strength,
                "ref_price": prev.close,
                "price_delta": delta,
                "atr_multiplier": dyn_multiplier,
            })
            confidence = min(1.2, 0.8 + 0.2 * strength)
            signals.append(self.make_signal(symbol, "LONG", confidence=confidence, metadata=meta_long))

        if delta < -threshold and meta["trend"] != "UP" and meta["rel_volume"] >= 1.05:
            strength = (-delta / atr_value) * meta["rel_volume"]
            meta_short = dict(meta)
            meta_short.update({
                "signal_strength": strength,
                "ref_price": prev.close,
                "price_delta": delta,
                "atr_multiplier": dyn_multiplier,
            })
            confidence = min(1.2, 0.8 + 0.2 * strength)
            signals.append(self.make_signal(symbol, "SHORT", confidence=confidence, metadata=meta_short))

        return signals


__all__ = ["VolatilityBreakoutStrategy"]
