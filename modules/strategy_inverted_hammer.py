
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


class InvertedHammerStrategy(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 60
        ema_fast = 20
        ema_slow = 50
        atr_len = 14
        shadow_mult = 1.6
        low_shadow_max = 0.5

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Inverted Hammer+", abbreviation="IHAM+", interval=self.Cfg.interval, lookback=self.Cfg.lookback)

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 6:
            return []
        latest = candles[-1]
        prev = candles[-2]
        closes = [c.close for c in candles]
        atr = _atr(candles, self.Cfg.atr_len)
        if atr <= 0:
            return []

        # Downtrend
        if not (_ema(closes, self.Cfg.ema_fast) < _ema(closes, self.Cfg.ema_slow)):
            return []

        body = _body(latest)
        upper_shadow = latest.high - max(latest.open, latest.close)
        lower_shadow = min(latest.open, latest.close) - latest.low

        cond_shape = upper_shadow >= self.Cfg.shadow_mult * body and lower_shadow <= self.Cfg.low_shadow_max * body
        cond_close = latest.close > latest.open  # bullish
        size_ok = max(body, upper_shadow) / atr >= 0.4

        if cond_shape and cond_close and size_ok:
            strength = (upper_shadow / atr)
            conf = _soft_confidence(strength)
            meta = {"open": latest.open, "close": latest.close, "high": latest.high, "upper_shadow": upper_shadow, "atr": atr}
            return [self.make_signal(symbol, "LONG", confidence=conf, metadata=meta)]
        return []

__all__ = ["InvertedHammerStrategy"]
