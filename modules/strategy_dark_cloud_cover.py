from __future__ import annotations
from typing import Iterable, Sequence, Dict
from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
from modules.indicators import ema, atr, base_metadata, passes_sanity

def _body(c: Kline) -> float:
    return abs(c.close - c.open)

def _is_bull(c: Kline) -> bool:
    return c.close > c.open

def _is_bear(c: Kline) -> bool:
    return c.close < c.open

def _last(values):
    return values[-1] if values else None

def _confidence(score: float, lo: float = 0.5, hi: float = 0.98) -> float:
    score = max(0.0, min(2.0, score)) / 2.0
    return lo + (hi - lo) * score

def _trend_ok(meta: Dict[str, float], want: str) -> bool:
    t = meta.get("trend", "FLAT")
    return t == want

def _near(value: float, target: float, tol: float) -> bool:
    if target == 0:
        return False
    return abs(value - target) / abs(target) <= tol


class DarkCloudCoverStrategy(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 80
        min_atr_pct = 0.0008
        min_rel_vol = 0.9

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="DarkCloudCover",
            abbreviation="DCC+",
            interval=self.Cfg.interval,
            lookback=self.Cfg.lookback,
            extra_timeframes={"30m": 120, "1h": 120}
        )

    def process(
        self,
        symbol: str,
        candles: Sequence[Kline],
    ) -> Iterable[Signal]:
        return self.process_with_timeframes(symbol, candles, {})

    def process_with_timeframes(
        self,
        symbol: str,
        primary_candles: Sequence[Kline],
        extra_candles: Dict[str, Sequence[Kline]],
    ) -> Iterable[Signal]:
        candles = primary_candles
        if len(candles) < self.Cfg.lookback:
            return []

        meta = base_metadata(candles)
        if not passes_sanity(
            meta,
            min_atr_pct=self.Cfg.min_atr_pct,
            min_rel_vol=self.Cfg.min_rel_vol,
        ):
            return []

        closes = [c.close for c in candles]
        e20 = _last(ema(closes, 20))
        e50 = _last(ema(closes, 50))
        atr_val = _last(atr(candles, 14)) or 0.0

        # Confirm higher timeframe trend
        trend_ok = _trend_ok(meta, "UP")

        if not trend_ok:
            return []

        previous = candles[-2]
        current = candles[-1]

        if not _is_bull(previous):
            return []

        body_previous = _body(previous)
        penetration_level = previous.open + body_previous * 0.5
        opens_near_high = current.open >= previous.high * 0.999
        closes_deep = current.close < penetration_level

        if (
            _is_bear(current)
            and opens_near_high
            and closes_deep
            and e20
            and e50
            and e20 > e50
        ):
            atr_safe = atr_val or 1e-9
            body_ratio = _body(current) / atr_safe

            if body_ratio >= 0.3:
                drop_strength = (previous.close - current.close) / atr_safe
                confidence = _confidence(drop_strength)
                signal = self.make_signal(
                    symbol,
                    "SHORT",
                    confidence=confidence,
                    metadata={"pattern": "dark_cloud_cover"},
                )
                return [signal]
        return []


__all__ = ["DarkCloudCoverStrategy"]
