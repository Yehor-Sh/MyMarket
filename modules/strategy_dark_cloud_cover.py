from __future__ import annotations
from typing import Iterable, Sequence, Dict
from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
from indicators import ema, atr, rsi, base_metadata, passes_sanity

# ---- Shared helpers for candle anatomy ----
def _body(c: Kline) -> float:
    return abs(c.close - c.open)

def _is_bull(c: Kline) -> bool:
    return c.close > c.open

def _is_bear(c: Kline) -> bool:
    return c.close < c.open

def _last(values):
    return values[-1] if values else None

def _confidence(score: float, lo: float = 0.5, hi: float = 0.98) -> float:
    # squash to [lo, hi] with soft cap
    score = max(0.0, min(2.0, score)) / 2.0
    return lo + (hi - lo) * score

def _trend_ok(meta: Dict[str, float], want: str) -> bool:
    t = meta.get("trend", "FLAT")
    return t == want

def _near(value: float, target: float, tol: float) -> bool:
    if target == 0:
        return False
    return abs(value - target) / abs(target) <= tol

class DarkCloudCoverStrategy(ModuleBase):
    class Cfg:
        interval = "15m"
        lookback = 80
        min_atr_pct = 0.0008
        min_rel_vol = 0.9
        # HTF filters
        rsi1h_min = 45 if "SHORT" == "LONG" else 0
        rsi1h_max = 100 if "SHORT" == "LONG" else 55
        ema_anchor_tol = 0.01  # 1% proximity penalty

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="DarkCloudCover",
            abbreviation="DCC+",
            interval=self.Cfg.interval,
            lookback=self.Cfg.lookback,
            extra_timeframes={"30m": 120, "1h": 120}
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        # Fallback path if extra TFs are not supplied by worker config
        return self.process_with_timeframes(symbol, candles, {})

    def process_with_timeframes(
        self,
        symbol: str,
        primary_candles: Sequence[Kline],
        extra_candles: Dict[str, Sequence[Kline]],
    ) -> Iterable[Signal]:
        candles = primary_candles
        if len(candles) < self.Cfg.lookback:
            return []

        meta = base_metadata(candles)
        if not passes_sanity(meta, min_atr_pct=self.Cfg.min_atr_pct, min_rel_vol=self.Cfg.min_rel_vol):
            return []

        last = candles[-1]
        closes = [c.close for c in candles]
        e20_series = ema(closes, 20)
        e50_series = ema(closes, 50)
        e200_series = ema(closes, 200)
        e20 = _last(e20_series)
        e50 = _last(e50_series)
        e200 = _last(e200_series)
        atr_val_series = atr(candles, 14)
        atr_val = _last(atr_val_series) or 0.0
        atr_pct = (atr_val / last.close) if last.close else 0.0

        # HTF filters
        rsi1h_ok = True
        trend_ok = True
        if extra_candles.get("1h"):
            rsi1h = _last(rsi([c.close for c in extra_candles["1h"]], 14)) or 50
            if "SHORT" == "LONG":
                rsi1h_ok = rsi1h >= self.Cfg.rsi1h_min
            else:
                rsi1h_ok = rsi1h <= self.Cfg.rsi1h_max
            # trend from 1h
            meta1h = base_metadata(extra_candles["1h"])
            trend_ok = _trend_ok(meta1h, "UP")
        else:
            trend_ok = _trend_ok(meta, "UP")

        if not (rsi1h_ok and trend_ok):
            return []

        # penalty if bouncing right into EMA200
        ema_anchor_penalty = 0.0
        if e200 and _near(last.close, e200, self.Cfg.ema_anchor_tol):
            ema_anchor_penalty = 0.15

        sig = (c1, c2 = candles[-2], candles[-1]
if not _is_bull(c1):
    return None
body1 = _body(c1)
pen_level = c1.open + body1*0.5
soft_high_ok = c2.open >= (c1.high * 0.999)  # 0.1% tolerance
closes_deep = _is_bear(c2) and c2.close < pen_level
size_ok = (_body(c2) / (atr_val or 1e-9)) >= 0.3
ema_up = e20 and e50 and e20 > e50
if soft_high_ok and closes_deep and size_ok and ema_up:
    strength = (c1.close - c2.close) / (atr_val or 1e-9)
    conf = _confidence(strength)
    return Signal(symbol=symbol, side="SHORT", strategy=self.abbreviation, confidence=conf, metadata={"pattern":"dark_cloud_cover"})
return None
)
        if sig is None:
            return []
        # Adjust confidence for environment
        sig_conf = sig.confidence * (1.0 - ema_anchor_penalty)
        meta_out = dict(sig.metadata or {})
        meta_out.update(meta)
        meta_out.update({"atr_value": atr_val, "atr_pct": atr_pct, "ema20": e20, "ema50": e50, "ema200": e200})
        return [Signal(symbol=symbol.upper(), side=sig.side, strategy=self.abbreviation, confidence=max(0.4, min(0.99, sig_conf)), metadata=meta_out)]

__all__ = ["DarkCloudCoverStrategy"]
