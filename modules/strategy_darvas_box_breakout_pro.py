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


def _box_high(cs: Sequence[Kline], n: int = 12) -> float:
    return max(c.high for c in cs[-n-1:-1]) if len(cs) > n else cs[-1].high
def _box_low(cs: Sequence[Kline], n: int = 12) -> float:
    return min(c.low for c in cs[-n-1:-1]) if len(cs) > n else cs[-1].low

class DarvasBoxBreakoutPro(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 160
        min_atr_pct = 0.0008
        min_rel_vol = 0.9
        trend_check = True
        box = 12

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Darvas Box Breakout Pro", abbreviation="DARX",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)

    def process(self, symbol: str, candles: Sequence[Kline]):
        if len(candles) < self.Cfg.lookback: return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol): return []
        last = candles[-1]; a = meta.get("atr") or 0.0
        hi = _box_high(candles, self.Cfg.box); lo = _box_low(candles, self.Cfg.box)
        if last.close > hi and _is_bull(last) and _body(last) >= 0.4*a and (not self.Cfg.trend_check or meta.get("trend") == "UP"):
            return [self.make_signal(symbol, "LONG", confidence=_conf((_body(last))/(a or 1e-9)), metadata={"pattern":"darvas_long"})]
        if last.close < lo and _is_bear(last) and _body(last) >= 0.4*a and (not self.Cfg.trend_check or meta.get("trend") == "DOWN"):
            return [self.make_signal(symbol, "SHORT", confidence=_conf((_body(last))/(a or 1e-9)), metadata={"pattern":"darvas_short"})]
        return []

__all__ = ["DarvasBoxBreakoutPro"]
