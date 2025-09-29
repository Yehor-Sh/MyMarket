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


class VWAPTrend4H(ModuleBase):
    class Cfg:
        interval = "4h"
        lookback = 120
        atr_len = 14
        min_atr_pct = 0.005
        min_rel_vol = 0.8

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="VWAPTrend4H", abbreviation="VWTC4H", interval=self.Cfg.interval, lookback=self.Cfg.lookback)

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

        vwap_val = _last(_rolling_vwap(candles, length=50))
        if vwap_val != vwap_val or vwap_val is None:
            return []
        long_ok  = _is_bull(last) and (last.close >= vwap_val + 1.0*atr_val)
        short_ok = _is_bear(last) and (last.close <= vwap_val - 1.0*atr_val)
        if long_ok:
            strength = ((last.close - vwap_val)/(atr_val or 1e-9)) + (1.0 if (e20 and e50 and e20 >= e50) else 0.0)
            conf = _confidence(strength)
            m = dict(meta); m.update({"pattern":"VWTC4H","vwap":vwap_val,"atr":atr_val,"unclosed":True})
            return [Signal(symbol=symbol, side="LONG", strategy=self.abbreviation, confidence=conf, metadata=m)]
        if short_ok:
            strength = ((vwap_val - last.close)/(atr_val or 1e-9)) + (1.0 if (e20 and e50 and e20 <= e50) else 0.0)
            conf = _confidence(strength)
            m = dict(meta); m.update({"pattern":"VWTC4H","vwap":vwap_val,"atr":atr_val,"unclosed":True})
            return [Signal(symbol=symbol, side="SHORT", strategy=self.abbreviation, confidence=conf, metadata=m)]
        return []


__all__ = ["VWAPTrend4H"]
