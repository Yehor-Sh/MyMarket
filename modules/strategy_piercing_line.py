
from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

# ---- Small utility helpers kept local to avoid extra dependencies ----
def _ema(values, period: int) -> float:
    if not values or period <= 1:
        return values[-1] if values else 0.0
    p = min(period, len(values))
    # seed with SMA
    sma = sum(values[-p:]) / p
    k = 2 / (p + 1)
    ema_val = sma
    for v in values[-p+1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val

def _atr(candles: Sequence[Kline], period: int) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        c = candles[i]
        prev = candles[i-1]
        tr = max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close))
        trs.append(tr)
    return sum(trs) / len(trs)

def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0: 
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid-1] + s[mid]) / 2.0

def _soft_confidence(strength: float, lo=0.55, hi=0.95) -> float:
    # squashed to [lo, hi]
    x = max(0.0, min(2.0, strength)) / 2.0
    return lo + (hi - lo) * x

def _body(c: Kline) -> float:
    return abs(c.close - c.open)

def _is_bull(c: Kline) -> bool:
    return c.close > c.open

def _is_bear(c: Kline) -> bool:
    return c.close < c.open


class PiercingLineStrategy(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 60
        ema_fast = 20
        ema_slow = 50
        atr_len = 14
        penetration = 0.6  # 60% into c1 body

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Piercing Line+", abbreviation="PLN+", interval=self.Cfg.interval, lookback=self.Cfg.lookback)

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 6:
            return []
        c1, c2 = candles[-2], candles[-1]
        closes = [c.close for c in candles]
        atr = _atr(candles, self.Cfg.atr_len)
        if atr <= 0:
            return []

        if not (_ema(closes, self.Cfg.ema_fast) < _ema(closes, self.Cfg.ema_slow)):
            return []

        if not _is_bear(c1):
            return []

        body1 = _body(c1)
        pierce_level = c1.open - body1 * (1 - self.Cfg.penetration)  # equivalently: c1.open - 0.4*body1 for 60%
        cond2 = _is_bull(c2) and c2.close >= pierce_level
        size_ok = _body(c2) / atr >= 0.35

        if cond2 and size_ok:
            strength = (_body(c2) / atr)
            conf = _soft_confidence(strength)
            meta = {"c1": (c1.open, c1.close), "c2": (c2.open, c2.close), "atr": atr}
            return [self.make_signal(symbol, "LONG", confidence=conf, metadata=meta)]
        return []

__all__ = ["PiercingLineStrategy"]
