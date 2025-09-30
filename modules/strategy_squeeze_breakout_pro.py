from __future__ import annotations
from typing import Iterable, Sequence, Dict, List
from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
try:
    from indicators import ema, atr, rsi, base_metadata, passes_sanity
except Exception:
    from modules.indicators import ema, atr, rsi, base_metadata, passes_sanity

def _last(v): return v[-1] if v else None
def _body(c: Kline) -> float: return abs(c.close - c.open)
def _is_bull(c: Kline) -> bool: return c.close > c.open
def _is_bear(c: Kline) -> bool: return c.close < c.open
def _conf(x, lo=0.55, hi=0.96):
    x = max(0.0, min(2.0, x)) / 2.0
    return lo + (hi - lo) * x


def _range(cs: Sequence[Kline], n: int) -> float:
    seg = cs[-n-1:-1] if len(cs) > n else cs[:-1]
    if not seg: return 0.0
    hi = max(c.high for c in seg); lo = min(c.low for c in seg)
    return hi - lo, hi, lo

class SqueezeBreakoutPro(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 160
        window = 18
        min_atr_pct = 0.0008
        min_rel_vol = 0.9
        trend_check = True

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Squeeze Breakout Pro", abbreviation="SQBX",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)

    def process(self, symbol: str, candles: Sequence[Kline]):
        if len(candles) < max(self.Cfg.lookback, self.Cfg.window+5): return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol): return []
        a = meta.get("atr") or 0.0
        width, hi, lo = _range(candles, self.Cfg.window)
        last = candles[-1]
        if a <= 0: return []
        if width <= 1.1*a:  # сжатие
            if last.close > hi and _is_bull(last) and _body(last) >= 0.4*a and (not self.Cfg.trend_check or meta.get("trend") == "UP"):
                return [self.make_signal(symbol, "LONG", confidence=_conf((_body(last))/(a or 1e-9)), metadata={"pattern":"squeeze_break_long"})]
            if last.close < lo and _is_bear(last) and _body(last) >= 0.4*a and (not self.Cfg.trend_check or meta.get("trend") == "DOWN"):
                return [self.make_signal(symbol, "SHORT", confidence=_conf((_body(last))/(a or 1e-9)), metadata={"pattern":"squeeze_break_short"})]
        return []

__all__ = ["SqueezeBreakoutPro"]
