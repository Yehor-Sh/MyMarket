from __future__ import annotations
from typing import Iterable, Sequence, Dict, List
from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
try:
    from indicators import ema, atr, rsi, base_metadata, passes_sanity
except Exception:
    from modules.indicators import ema, atr, rsi, base_metadata, passes_sanity

def _body(c: Kline) -> float: return abs(c.close - c.open)
def _is_bull(c: Kline) -> bool: return c.close > c.open
def _is_bear(c: Kline) -> bool: return c.close < c.open
def _last(v): return v[-1] if v else None
def _conf(x, lo=0.5, hi=0.99):
    x = max(0.0, min(2.0, x)) / 2.0
    return lo + (hi - lo) * x
def _n_high(cs: Sequence[Kline], n: int) -> float:
    import math
    return max(c.high for c in cs[-n-1:-1]) if len(cs) > n else math.nan
def _n_low(cs: Sequence[Kline], n: int) -> float:
    import math
    return min(c.low for c in cs[-n-1:-1]) if len(cs) > n else math.nan
def _vwap_series(cs: Sequence[Kline], win: int = 50) -> List[float]:
    out: List[float] = []
    pv = vv = 0.0
    fifo: List[tuple[float,float]] = []
    for c in cs:
        tp = (c.high + c.low + c.close) / 3.0
        pvt = tp * (c.volume or 0.0); v = (c.volume or 0.0)
        pv += pvt; vv += v
        fifo.append((pvt, v))
        if len(fifo) > win:
            op, ov = fifo.pop(0); pv -= op; vv -= ov
        out.append((pv / vv) if vv else float("nan"))
    return out


class ReversalPin4H(ModuleBase):
    class Cfg:
        interval = "4h"
        lookback = 120
        atr_len = 14
        min_atr_pct = 0.002
        min_rel_vol = 0.5

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="ReversalPin4H", abbreviation="RPIN4H",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback)

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < self.Cfg.lookback:
            return []
        meta = base_metadata(candles, atr_period=self.Cfg.atr_len)
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol):
            return []
        last = candles[-1]
        closes = [c.close for c in candles]
        atr_val = _last(atr(candles, self.Cfg.atr_len)) or 0.0
        e20 = _last(ema(closes, 20)); e50 = _last(ema(closes, 50))
        body = _body(last); upper = last.high - max(last.open, last.close); lower = min(last.open, last.close) - last.low
        hi = _n_high(candles, 20); lo = _n_low(candles, 20)
        if (lower >= 1.5*body) and (lower >= 0.5*atr_val) and (last.low <= lo*1.001) and _is_bull(last):
            m = dict(meta); m.update({"pattern":"RPIN4H","level":lo,"atr":atr_val,"unclosed":True})
            return [self.make_signal(symbol, "LONG", confidence=_conf(lower/(atr_val or 1e-9)), metadata=m)]
        if (upper >= 1.5*body) and (upper >= 0.5*atr_val) and (last.high >= hi*0.999) and _is_bear(last):
            m = dict(meta); m.update({"pattern":"RPIN4H","level":hi,"atr":atr_val,"unclosed":True})
            return [self.make_signal(symbol, "SHORT", confidence=_conf(upper/(atr_val or 1e-9)), metadata=m)]
        return []


__all__ = ["ReversalPin4H"]
