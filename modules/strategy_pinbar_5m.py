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

class PinBar5m(ModuleBase):
    class Cfg:
        interval = "5m"; lookback = 60; min_atr_pct = 0.0006; min_rel_vol = 0.7
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Pin Bar 5m", abbreviation="PIN5",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)
    def process(self, symbol: str, candles: Sequence[Kline]):
        if len(candles) < self.Cfg.lookback: return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, self.Cfg.min_atr_pct, self.Cfg.min_rel_vol): return []
        a = meta.get("atr") or 0.0; c = candles[-1]; body = _body(c)
        upper = c.high - max(c.open, c.close); lower = min(c.open, c.close) - c.low
        if lower >= 1.5*body and lower >= 0.4*a and _is_bull(c):
            return [self.make_signal(symbol, "LONG", confidence=_conf(lower/(a or 1e-9)), metadata={"pattern":"pinbar5_long"})]
        if upper >= 1.5*body and upper >= 0.4*a and _is_bear(c):
            return [self.make_signal(symbol, "SHORT", confidence=_conf(upper/(a or 1e-9)), metadata={"pattern":"pinbar5_short"})]
        return []
__all__ = ["PinBar5m"]
