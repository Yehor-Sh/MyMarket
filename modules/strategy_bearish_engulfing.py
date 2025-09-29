from __future__ import annotations
from typing import Iterable, Sequence, Dict
from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
from modules.indicators import ema, atr, rsi, base_metadata, passes_sanity

def _body(c: Kline) -> float:
    return abs(c.close - c.open)

def _is_bull(c: Kline) -> bool:
    return c.close > c.open

def _is_bear(c: Kline) -> bool:
    return c.close < c.open

def _last(values):
    return values[-1] if values else None

def _confidence(score: float, lo: float = 0.4, hi: float = 0.99) -> float:
    score = max(0.0, min(2.0, score)) / 2.0
    return lo + (hi - lo) * score


class BearishEngulfingStrategy(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 60
        min_atr_pct = 0.0005   # ослаблен
        min_rel_vol = 0.7      # ослаблен

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="BearishEngulfing",
            abbreviation="BENGF+",
            interval=self.Cfg.interval,
            lookback=self.Cfg.lookback,
            extra_timeframes={"1h": 100}
        )

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
        e20 = _last(ema(closes, 20))
        e50 = _last(ema(closes, 50))
        atr_val = _last(atr(candles, 14)) or 0.0

        c1, c2 = candles[-2], candles[-1]
        if _is_bull(c1) and _is_bear(c2) and c2.close <= c1.open:
            if (_body(c2)/(atr_val or 1e-9)) >= 0.25 and e20 and e50 and e20 < e50:
                conf=_confidence(_body(c2)/(atr_val or 1e-9))
                return [self.make_signal(symbol,"SHORT",confidence=conf,metadata={"pattern":"bearish_engulfing"})]
        return []


__all__ = ["BearishEngulfingStrategy"]
