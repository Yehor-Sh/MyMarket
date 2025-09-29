from __future__ import annotations
from typing import Iterable, Sequence, Dict
from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
from modules.indicators import atr, base_metadata, passes_sanity

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


class ThreeWhiteSoldiersStrategy(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 80
        min_atr_pct = 0.0008
        min_rel_vol = 0.9

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="ThreeWhiteSoldiers",
            abbreviation="TWS+",
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

        atr_val = _last(atr(candles, 14)) or 0.0

        # Confirm higher timeframe trend
        trend_ok = _trend_ok(meta, "DOWN")

        if not trend_ok:
            return []

        first = candles[-3]
        second = candles[-2]
        third = candles[-1]

        atr_safe = atr_val or 1e-9
        tolerance = (atr_val or 0.0) * 0.25
        average_body = (
            _body(first) + _body(second) + _body(third)
        ) / 3
        body_ratio = average_body / atr_safe

        second_within_range = (
            min(first.open, first.close) - tolerance
            <= second.open
            <= max(first.open, first.close) + tolerance
        )
        third_within_range = (
            min(second.open, second.close) - tolerance
            <= third.open
            <= max(second.open, second.close) + tolerance
        )

        if (
            _is_bull(first)
            and _is_bull(second)
            and _is_bull(third)
            and second.close > first.close
            and third.close > second.close
            and second_within_range
            and third_within_range
            and body_ratio >= 0.28
        ):
            confidence = _confidence(body_ratio)
            signal = self.make_signal(
                symbol,
                "LONG",
                confidence=confidence,
                metadata={"pattern": "three_white_soldiers"},
            )
            return [signal]
        return []


__all__ = ["ThreeWhiteSoldiersStrategy"]
