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


class ThreeSoldiersCrowsPro(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 140
        min_atr_pct = 0.0008
        min_rel_vol = 0.9
        trend_check = True

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Three Soldiers/Crows Pro", abbreviation="TSCR",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)

    def process(self, symbol: str, candles: Sequence[Kline]):
        if len(candles) < self.Cfg.lookback: return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol): return []
        a = meta.get("atr") or 0.0
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if _is_bull(c1) and _is_bull(c2) and _is_bull(c3) and (_body(c1)+_body(c2)+_body(c3))/3.0 >= 0.45*a:
            if not self.Cfg.trend_check or meta.get("trend") == "UP":
                return [self.make_signal(symbol, "LONG", confidence=_conf((_body(c3))/(a or 1e-9)), metadata={"pattern":"three_white_soldiers"})]
        if _is_bear(c1) and _is_bear(c2) and _is_bear(c3) and (_body(c1)+_body(c2)+_body(c3))/3.0 >= 0.45*a:
            if not self.Cfg.trend_check or meta.get("trend") == "DOWN":
                return [self.make_signal(symbol, "SHORT", confidence=_conf((_body(c3))/(a or 1e-9)), metadata={"pattern":"three_black_crows"})]
        return []

__all__ = ["ThreeSoldiersCrowsPro"]
