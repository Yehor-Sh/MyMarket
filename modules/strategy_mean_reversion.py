"""Mean reversion strategy with z-score and reversal confirmation."""


from __future__ import annotations

from statistics import fmean, pstdev
from typing import Iterable, Sequence


from binance_client import Kline
from module_base import ModuleBase, Signal

from .strategy_shared import base_metadata, passes_sanity


class MeanReversionStrategy(ModuleBase):
    """Fade extremes only against stretched z-score with reversal confirmation."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 280,
        window: int = 24,
        z_threshold: float = 2.0,
    ) -> None:
        minimum_history = max(lookback, window + 220, 240)

        super().__init__(
            client,
            name="MeanReversion",
            abbreviation="MRV",
            interval=interval,
            lookback=minimum_history,
        )
        self._window = window
        self._z_threshold = z_threshold

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        bars = list(candles)
        if len(bars) <= self._window:
            return []

        meta = base_metadata(bars)
        if not passes_sanity(meta, min_atr_pct=0.0006, min_rel_vol=0.8):
            return []

        closes = [c.close for c in bars]
        recent = closes[-self._window :]
        mean = fmean(recent)
        std = pstdev(recent) if len(recent) > 1 else 0.0
        if std == 0:
            return []

        z_score = (closes[-1] - mean) / std

        prev = bars[-2]
        last = bars[-1]

        bull_reversal = (
            z_score <= -self._z_threshold
            and last.close > last.open
            and prev.close < prev.open
            and meta["trend"] != "DOWN"
        )
        bear_reversal = (
            z_score >= self._z_threshold
            and last.close < last.open
            and prev.close > prev.open
            and meta["trend"] != "UP"
        )

        meta["ref_price"] = mean

        signals: list[Signal] = []
        if bull_reversal:
            strength = (-z_score) * (1.0 + meta["atr_pct"] * 40) * max(1.0, meta["rel_volume"])
            meta_long = dict(meta)
            meta_long.update({
                "signal_strength": strength,
                "z_score": z_score,
                "mean": mean,
                "std": std,
            })
            confidence = min(1.15, 0.75 + 0.2 * strength)
            signals.append(self.make_signal(symbol, "LONG", confidence=confidence, metadata=meta_long))

        if bear_reversal:
            strength = (z_score) * (1.0 + meta["atr_pct"] * 40) * max(1.0, meta["rel_volume"])
            meta_short = dict(meta)
            meta_short.update({
                "signal_strength": strength,
                "z_score": z_score,
                "mean": mean,
                "std": std,
            })
            confidence = min(1.15, 0.75 + 0.2 * strength)
            signals.append(self.make_signal(symbol, "SHORT", confidence=confidence, metadata=meta_short))

        return signals


__all__ = ["MeanReversionStrategy"]
