from __future__ import annotations
from typing import Iterable, Sequence, Dict
from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
from modules.indicators import ema, atr, rsi, base_metadata, passes_sanity

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

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
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
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol):
            return []

        last = candles[-1]
        closes = [c.close for c in candles]
        e20 = _last(ema(closes, 20))
        e50 = _last(ema(closes, 50))
        e200 = _last(ema(closes, 200))
        atr_val = _last(atr(candles, 14)) or 0.0

        # Confirm higher timeframe trend
        trend_ok = _trend_ok(meta, "DOWN")

        if not trend_ok:
            return []

        c1,c2,c3=candles[-3],candles[-2],candles[-1]
        if _is_bull(c1) and _is_bull(c2) and _is_bull(c3):
            if c2.close>c1.close and c3.close>c2.close:
                avg_body=(_body(c1)+_body(c2)+_body(c3))/3
                tol=(atr_val or 0.0)*0.25
                within2=min(c1.open,c1.close)-tol<=c2.open<=max(c1.open,c1.close)+tol
                within3=min(c2.open,c2.close)-tol<=c3.open<=max(c2.open,c2.close)+tol
                if avg_body/(atr_val or 1e-9)>=0.28 and within2 and within3:
                    strength=avg_body/(atr_val or 1e-9)
                    conf=_confidence(strength)
                    return [self.make_signal(symbol,"LONG",confidence=conf,metadata={"pattern":"three_white_soldiers"})]
        return []


__all__ = ["ThreeWhiteSoldiersStrategy"]
