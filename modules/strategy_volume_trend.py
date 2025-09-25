"""Volume weighted trend strategy tuned for short-term impulses."""

from __future__ import annotations

from typing import Iterable, Sequence

from binance_client import Kline
from module_base import ModuleBase, Signal

from .strategy_shared import base_metadata, passes_sanity


class VolumeWeightedTrendStrategy(ModuleBase):
    """5m trend-following with relative volume and ATR-normalised impulse."""

    def __init__(
        self,
        client,
        *,
        interval: str = "5m",
        lookback: int = 240,
        volume_window: int = 20,
        momentum_window: int = 6,
        impulse_min_atr: float = 0.25,
    ) -> None:
        minimum_history = max(lookback, volume_window + 210, momentum_window + 210, 220)
        super().__init__(
            client,
            name="VolumeWeightedTrend",
            abbreviation="VWT",
            interval=interval,
            lookback=minimum_history,
        )
        self._volume_window = volume_window
        self._momentum_window = momentum_window
        self._impulse_min_atr = impulse_min_atr

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        bars = list(candles)
        if len(bars) <= max(self._volume_window, self._momentum_window):
            return []

        meta = base_metadata(bars, vol_window=self._volume_window)
        if not passes_sanity(meta, min_atr_pct=0.0007, min_rel_vol=1.05):
            return []

        closes = [c.close for c in bars]
        last_close = closes[-1]
        reference_close = closes[-self._momentum_window]
        atr_value = meta["atr"]
        if atr_value <= 0:
            return []

        impulse_atr = (last_close - reference_close) / atr_value
        meta["ref_price"] = reference_close

        signals: list[Signal] = []
        if impulse_atr >= self._impulse_min_atr and meta["trend"] == "UP":
            meta_long = dict(meta)
            meta_long.update({
                "signal_strength": impulse_atr,
                "momentum_reference": reference_close,
            })
            confidence = min(1.25, 0.85 + 0.2 * impulse_atr) * max(1.0, meta_long["rel_volume"])
            signals.append(self.make_signal(symbol, "LONG", confidence=confidence, metadata=meta_long))
        elif impulse_atr <= -self._impulse_min_atr and meta["trend"] == "DOWN":
            meta_short = dict(meta)
            meta_short.update({
                "signal_strength": -impulse_atr,
                "momentum_reference": reference_close,
            })
            confidence = min(1.25, 0.85 + 0.2 * (-impulse_atr)) * max(1.0, meta_short["rel_volume"])
            signals.append(self.make_signal(symbol, "SHORT", confidence=confidence, metadata=meta_short))

        return signals


__all__ = ["VolumeWeightedTrendStrategy"]
