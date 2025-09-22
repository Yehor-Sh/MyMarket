"""Momentum factor strategy measuring percentage change over N bars."""

from __future__ import annotations

from typing import Iterable, Sequence


from binance_client import Kline
from module_base import ModuleBase, Signal

from .strategy_shared import base_metadata, passes_sanity

class MomentumFactorStrategy(ModuleBase):
    """Capture short term momentum bursts in the direction of the trend."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 240,
        momentum_window: int = 10,
        threshold: float = 0.008,
    ) -> None:
        minimum_history = max(lookback, momentum_window + 210, 220)

        super().__init__(
            client,
            name="MomentumFactor",
            abbreviation="MOM",
            interval=interval,
            lookback=minimum_history,
        )
        self._momentum_window = momentum_window
        self._threshold = threshold

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        bars = list(candles)
        if len(bars) <= self._momentum_window:
            return []

        closes = [c.close for c in bars]
        current_close = closes[-1]
        reference_close = closes[-self._momentum_window]
        if reference_close == 0:
            return []

        ret = (current_close / reference_close) - 1.0
        meta = base_metadata(bars)
        if not passes_sanity(meta):
            return []

        meta["ref_price"] = reference_close
        aligned_long = meta["trend"] == "UP"
        aligned_short = meta["trend"] == "DOWN"

        strength = abs(ret) / max(self._threshold, 1e-9)

        signals: list[Signal] = []
        if ret >= self._threshold and aligned_long:
            meta_long = dict(meta)
            meta_long.update({
                "signal_strength": strength,
                "momentum_return": ret,
            })
            confidence = min(1.2, 0.7 + 0.3 * strength) * max(1.0, meta_long["rel_volume"])
            signals.append(self.make_signal(symbol, "LONG", confidence=confidence, metadata=meta_long))
        if ret <= -self._threshold and aligned_short:
            meta_short = dict(meta)
            meta_short.update({
                "signal_strength": strength,
                "momentum_return": ret,
            })
            confidence = min(1.2, 0.7 + 0.3 * strength) * max(1.0, meta_short["rel_volume"])
            signals.append(self.make_signal(symbol, "SHORT", confidence=confidence, metadata=meta_short))

        return signals


__all__ = ["MomentumFactorStrategy"]
