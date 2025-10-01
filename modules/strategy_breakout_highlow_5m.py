from __future__ import annotations
from typing import Sequence
from binance_client import BinanceClient, Kline
from module_base import ModuleBase
try:
    from indicators import ema, atr, base_metadata, passes_sanity
except Exception:
    from modules.indicators import ema, atr, base_metadata, passes_sanity

def _last(v): return v[-1] if v else None
def _body(c: Kline) -> float: return abs(c.close - c.open)
def _is_bull(c: Kline) -> bool: return c.close > c.open
def _is_bear(c: Kline) -> bool: return c.close < c.open
def _conf(x, lo=0.55, hi=0.9):
    x = max(0.0, min(2.0, x)) / 2.0
    return lo + (hi - lo) * x
def passes_sanity(meta, min_atr_pct=0.0, min_rel_vol=0.0):
    return meta.get("atr_pct", 0) >= min_atr_pct and meta.get("rel_volume", 0) >= min_rel_vol

def _n_high(cs: Sequence[Kline], n: int) -> float:
    return max(c.high for c in cs[-n-1:-1]) if len(cs) > n else cs[-1].high
def _n_low(cs: Sequence[Kline], n: int) -> float:
    return min(c.low for c in cs[-n-1:-1]) if len(cs) > n else cs[-1].low
class BreakoutHighLow5m(ModuleBase):
    class Cfg:
        interval = "5m"; lookback = 70; min_atr_pct = 0.0006; min_rel_vol = 0.7; window = 10
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Breakout HighLow 5m", abbreviation="BRK5",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)
    def process(self, symbol: str, candles: Sequence[Kline]):
        if len(candles) < max(self.Cfg.lookback, self.Cfg.window+5): return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, self.Cfg.min_atr_pct, self.Cfg.min_rel_vol): return []
        last = candles[-1]; a = meta.get("atr") or 0.0
        hi = _n_high(candles, self.Cfg.window); lo = _n_low(candles, self.Cfg.window)
        if last.close > hi and _is_bull(last) and _body(last) >= 0.3*a:
            return [self.make_signal(symbol, "LONG", confidence=_conf(_body(last)/(a or 1e-9)), metadata={"pattern":"breakout5_long"})]
        if last.close < lo and _is_bear(last) and _body(last) >= 0.3*a:
            return [self.make_signal(symbol, "SHORT", confidence=_conf(_body(last)/(a or 1e-9)), metadata={"pattern":"breakout5_short"})]
        return []
__all__ = ["BreakoutHighLow5m"]
