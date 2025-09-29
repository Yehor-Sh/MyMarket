from __future__ import annotations
from typing import Iterable, Sequence, Dict, List
from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
try:
    from indicators import ema, atr, rsi, base_metadata, passes_sanity
except Exception:
    from modules.indicators import ema, atr, rsi, base_metadata, passes_sanity

def _body(c: Kline) -> float:
    return abs(c.close - c.open)

def _is_bull(c: Kline) -> bool:
    return c.close > c.open

def _is_bear(c: Kline) -> bool:
    return c.close < c.open

def _last(values: Sequence[float] | None):
    if not values:
        return None
    return values[-1]

def _confidence(score: float, lo: float = 0.5, hi: float = 0.99) -> float:
    s = max(0.0, min(2.0, score)) / 2.0
    return lo + (hi - lo) * s

def _rolling_vwap(candles: Sequence[Kline], length: int = 50) -> List[float]:
    out: List[float] = []
    pv_sum = 0.0
    vol_sum = 0.0
    fifo: List[tuple[float,float]] = []
    for c in candles:
        tp = (c.high + c.low + c.close) / 3.0
        pv = tp * (c.volume or 0.0)
        v = c.volume or 0.0
        pv_sum += pv
        vol_sum += v
        fifo.append((pv, v))
        if len(fifo) > length:
            opv, ov = fifo.pop(0)
            pv_sum -= opv
            vol_sum -= ov
        out.append((pv_sum / vol_sum) if vol_sum else float('nan'))
    return out

def _n_high(candles: Sequence[Kline], n: int) -> float:
    if len(candles) <= n:
        return float('nan')
    return max(c.high for c in candles[-n-1:-1])

def _n_low(candles: Sequence[Kline], n: int) -> float:
    if len(candles) <= n:
        return float('nan')
    return min(c.low for c in candles[-n-1:-1])


class ReversalPin4H(ModuleBase):
    class Cfg:
        interval = "4h"
        lookback = 120
        atr_len = 14
        min_atr_pct = 0.005
        min_rel_vol = 0.8

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="ReversalPin4H", abbreviation="RPIN4H", interval=self.Cfg.interval, lookback=self.Cfg.lookback)

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < self.Cfg.lookback:
            return []
        meta = base_metadata(candles, atr_period=self.Cfg.atr_len)
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol):
            return []

        last = candles[-1]
        closes = [c.close for c in candles]
        atr_val = _last(atr(candles, self.Cfg.atr_len)) or 0.0
        e20 = _last(ema(closes, 20))
        e50 = _last(ema(closes, 50))

        body = _body(last)
        upper = last.high - max(last.open, last.close)
        lower = min(last.open, last.close) - last.low
        high20 = _n_high(candles, 20)
        low20  = _n_low(candles, 20)
        pin_top = (upper >= 2.0*body) and (upper >= 1.0*atr_val) and (last.high >= high20*0.999)
        pin_bot = (lower >= 2.0*body) and (lower >= 1.0*atr_val) and (last.low  <= low20*1.001)
        if pin_bot and _is_bull(last):
            conf = _confidence(lower/(atr_val or 1e-9))
            m = dict(meta); m.update({"pattern":"RPIN4H","atr":atr_val,"level":low20,"unclosed":True})
            return [Signal(symbol=symbol, side="LONG", strategy=self.abbreviation, confidence=conf, metadata=m)]
        if pin_top and _is_bear(last):
            conf = _confidence(upper/(atr_val or 1e-9))
            m = dict(meta); m.update({"pattern":"RPIN4H","atr":atr_val,"level":high20,"unclosed":True})
            return [Signal(symbol=symbol, side="SHORT", strategy=self.abbreviation, confidence=conf, metadata=m)]
        return []


__all__ = ["ReversalPin4H"]
