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

class Engulfing5m(ModuleBase):
    class Cfg:
        interval = "5m"; lookback = 60; min_atr_pct = 0.0006; min_rel_vol = 0.7
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Engulfing 5m", abbreviation="ENG5",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)
    def process(self, symbol: str, candles: Sequence[Kline]):
        if len(candles) < self.Cfg.lookback: return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, self.Cfg.min_atr_pct, self.Cfg.min_rel_vol): return []
        c1, c2 = candles[-2], candles[-1]; a = meta.get("atr") or 0.0
        if _is_bear(c1) and _is_bull(c2) and c2.close >= c1.open and c2.open <= c1.close and _body(c2) >= 0.3*a:
            return [self.make_signal(symbol, "LONG", confidence=_conf(_body(c2)/(a or 1e-9)), metadata={"pattern":"engulfing5_bull"})]
        if _is_bull(c1) and _is_bear(c2) and c2.open >= c1.close and c2.close <= c1.open and _body(c2) >= 0.3*a:
            return [self.make_signal(symbol, "SHORT", confidence=_conf(_body(c2)/(a or 1e-9)), metadata={"pattern":"engulfing5_bear"})]
        return []
__all__ = ["Engulfing5m"]
