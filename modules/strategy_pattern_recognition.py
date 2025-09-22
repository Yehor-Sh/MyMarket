"""Pattern based strategy focusing on two-candle reversals with filters."""

from __future__ import annotations

from typing import Iterable, Sequence

from binance_client import Kline
from module_base import ModuleBase, Signal

from .strategy_shared import base_metadata, passes_sanity


class PatternRecognitionStrategy(ModuleBase):
    """Two-candle reversals with body/volume/ATR guardrails and trend gating."""

    def __init__(
        self,
        client,
        *,
        interval: str = "15m",
        lookback: int = 240,
        body_frac_min: float = 0.45,
    ) -> None:
        minimum_history = max(lookback, 220)
        super().__init__(
            client,
            name="PatternRecognition",
            abbreviation="PAT",
            interval=interval,
            lookback=minimum_history,
        )
        self._body_frac_min = body_frac_min

    @staticmethod
    def _body_fraction(candle: Kline) -> float:
        price_range = max(candle.high - candle.low, 1e-12)
        return abs(candle.close - candle.open) / price_range

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        bars = list(candles)
        if len(bars) < 3:
            return []

        prev = bars[-2]
        curr = bars[-1]

        meta = base_metadata(bars)
        if not passes_sanity(meta):
            return []

        curr_body_ok = self._body_fraction(curr) >= self._body_frac_min
        if not curr_body_ok or meta["rel_volume"] < 1.1:
            return []

        meta["ref_price"] = prev.close

        signals: list[Signal] = []

        bull = (
            prev.close < prev.open
            and curr.close > curr.open
            and curr.close > prev.close
            and curr.close > meta["ema_fast"]
            and meta["trend"] != "DOWN"
        )
        bear = (
            prev.close > prev.open
            and curr.close < curr.open
            and curr.close < prev.close
            and curr.close < meta["ema_fast"]
            and meta["trend"] != "UP"
        )

        if bull:
            strength = self._body_fraction(curr) * meta["rel_volume"] * (1.0 + meta["atr_pct"] * 50)
            meta_long = dict(meta)
            meta_long.update({
                "signal_strength": strength,
                "reversal_type": "bullish",
            })
            confidence = min(1.2, 0.8 + 0.2 * strength)
            signals.append(self.make_signal(symbol, "LONG", confidence=confidence, metadata=meta_long))

        if bear:
            strength = self._body_fraction(curr) * meta["rel_volume"] * (1.0 + meta["atr_pct"] * 50)
            meta_short = dict(meta)
            meta_short.update({
                "signal_strength": strength,
                "reversal_type": "bearish",
            })
            confidence = min(1.2, 0.8 + 0.2 * strength)
            signals.append(self.make_signal(symbol, "SHORT", confidence=confidence, metadata=meta_short))

        return signals


__all__ = ["PatternRecognitionStrategy"]
