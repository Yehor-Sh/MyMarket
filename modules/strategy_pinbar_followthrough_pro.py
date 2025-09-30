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


class PinBarFollowThroughPro(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 140
        min_atr_pct = 0.0008
        min_rel_vol = 0.9
        trend_check = True

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="Pin Bar Follow-Through Pro", abbreviation="PINF",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)

    def process(self, symbol: str, candles: Sequence[Kline]):
        if len(candles) < self.Cfg.lookback: return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol): return []
        a = meta.get("atr") or 0.0
        pb, ft = candles[-2], candles[-1]
        body = _body(pb)
        upper = pb.high - max(pb.open, pb.close)
        lower = min(pb.open, pb.close) - pb.low
        # bullish pin + follow-through up
        if lower >= 1.8*body and lower >= 0.5*a and _is_bull(ft) and ft.close > max(pb.open, pb.close):
            if not self.Cfg.trend_check or meta.get("trend") == "UP":
                return [self.make_signal(symbol, "LONG", confidence=_conf((lower)/(a or 1e-9)), metadata={"pattern":"pinbar_follow_long"})]
        # bearish pin + follow-through down
        if upper >= 1.8*body and upper >= 0.5*a and _is_bear(ft) and ft.close < min(pb.open, pb.close):
            if not self.Cfg.trend_check or meta.get("trend") == "DOWN":
                return [self.make_signal(symbol, "SHORT", confidence=_conf((upper)/(a or 1e-9)), metadata={"pattern":"pinbar_follow_short"})]
        return []

__all__ = ["PinBarFollowThroughPro"]
