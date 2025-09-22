"""Range breakout strategy with volatility and volume confirmation."""

from __future__ import annotations

from typing import Iterable, Sequence

from binance_client import Kline
from module_base import ModuleBase, Signal

from .strategy_shared import base_metadata, passes_sanity


class RangeBreakoutStrategy(ModuleBase):
    """Breakout of recent range with volume + ATR confirmation and trend gating."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 240,
        breakout_window: int = 20,
        volume_window: int = 20,
        volume_multiplier: float = 1.3,
        min_atr_pct: float = 0.001,
    ) -> None:
        minimum_history = max(lookback, breakout_window + 210, volume_window + 210, 220)
        super().__init__(
            client,
            name="RangeBreakout",
            abbreviation="RBO",
            interval=interval,
            lookback=minimum_history,
        )
        self._breakout_window = breakout_window
        self._volume_window = volume_window
        self._volume_multiplier = volume_multiplier
        self._min_atr_pct = min_atr_pct

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        bars = list(candles)
        if len(bars) < self._breakout_window + 2:
            return []

        meta = base_metadata(bars, vol_window=self._volume_window)
        if not passes_sanity(meta, min_atr_pct=self._min_atr_pct, min_rel_vol=1.0):
            return []

        window = bars[-(self._breakout_window + 1) : -1]
        resistance = max(c.high for c in window)
        support = min(c.low for c in window)
        prev = bars[-2]
        last = bars[-1]

        rel_volume = meta["rel_volume"]

        long_ok = (
            last.close > resistance
            and prev.close <= resistance
            and rel_volume >= self._volume_multiplier
            and meta["trend"] != "DOWN"
        )
        short_ok = (
            last.close < support
            and prev.close >= support
            and rel_volume >= self._volume_multiplier
            and meta["trend"] != "UP"
        )

        signals: list[Signal] = []
        if long_ok:
            strength = (last.close - resistance) / max(meta["atr"], 1e-9) + (rel_volume - 1.0)
            meta_long = dict(meta)
            meta_long.update({
                "signal_strength": strength,
                "breakout_level": resistance,
            })
            meta_long["ref_price"] = resistance
            confidence = min(1.25, 0.85 + 0.25 * max(0.0, strength))
            signals.append(self.make_signal(symbol, "LONG", confidence=confidence, metadata=meta_long))

        if short_ok:
            strength = (support - last.close) / max(meta["atr"], 1e-9) + (rel_volume - 1.0)
            meta_short = dict(meta)
            meta_short.update({
                "signal_strength": strength,
                "breakdown_level": support,
            })
            meta_short["ref_price"] = support
            confidence = min(1.25, 0.85 + 0.25 * max(0.0, strength))
            signals.append(self.make_signal(symbol, "SHORT", confidence=confidence, metadata=meta_short))

        return signals


__all__ = ["RangeBreakoutStrategy"]
