from __future__ import annotations
from typing import Iterable, Sequence, Dict
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
def _conf(x, lo=0.5, hi=0.98):
    x = max(0.0, min(2.0, x)) / 2.0
    return lo + (hi - lo) * x


class MorningStarStrategy(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 70
        min_atr_pct = 0.0006
        min_rel_vol = 0.8

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(client, name="MorningStarStrategy", abbreviation="MSTAR+",
                         interval=self.Cfg.interval, lookback=self.Cfg.lookback,
                         extra_timeframes={"1h": 120})

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        return self.process_with_timeframes(symbol, candles, {})

    def process_with_timeframes(self, symbol: str, primary_candles: Sequence[Kline], extra_candles: Dict[str, Sequence[Kline]]) -> Iterable[Signal]:
        candles = primary_candles
        if len(candles) < self.Cfg.lookback:
            return []
        meta = base_metadata(candles)
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol):
            return []
        closes = [c.close for c in candles]
        e20 = _last(ema(closes, 20)); e50 = _last(ema(closes, 50))
        atr_val = _last(atr(candles, 14)) or 0.0
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if _is_bear(c1) and _body(c2) <= _body(c1)*0.6 and _is_bull(c3) and c3.close > (c1.open+c1.close)/2 and c3.close > c2.close and e20 and e50 and e20 < e50 and (_body(c3)/(atr_val or 1e-9)) >= 0.3:
            return [self.make_signal(symbol, "LONG", confidence=_conf(_body(c3)/(atr_val or 1e-9)), metadata={"pattern":"morning_star"})]
        return []


__all__ = ["MorningStarStrategy"]
