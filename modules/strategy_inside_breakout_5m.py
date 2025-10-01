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

class InsideBreakout5m(ModuleBase):
    class Cfg:
        interval = "5m"; lookback = 60; min_atr_pct = 0.0006; min_rel_vol = 0.7
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Inside Breakout 5m", abbreviation="INS5",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)
    def process(self, symbol: str, candles: Sequence[Kline]):
        if len(candles) < self.Cfg.lookback: return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, self.Cfg.min_atr_pct, self.Cfg.min_rel_vol): return []
        m, ib, br = candles[-3], candles[-2], candles[-1]; a = meta.get("atr") or 0.0
        if ib.high <= m.high and ib.low >= m.low:
            if br.close > m.high and _is_bull(br) and _body(br) >= 0.3*a:
                return [self.make_signal(symbol, "LONG", confidence=_conf(_body(br)/(a or 1e-9)), metadata={"pattern":"inside5_long"})]
            if br.close < m.low and _is_bear(br) and _body(br) >= 0.3*a:
                return [self.make_signal(symbol, "SHORT", confidence=_conf(_body(br)/(a or 1e-9)), metadata={"pattern":"inside5_short"})]
        return []
__all__ = ["InsideBreakout5m"]
