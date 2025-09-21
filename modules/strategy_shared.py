"""Shared helpers for advanced strategy metadata and filtering."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Dict, Sequence

from binance_client import Kline
from indicators import atr, ema


def _last_valid(values: Sequence[float]) -> float:
    for value in reversed(values):
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return float(value)
    return float("nan")


def trend_label(close: float, e20: float, e50: float, e200: float) -> str:
    if any(math.isnan(v) for v in (e20, e50, e200)):
        return "FLAT"
    if close > e50 and e50 > e200:
        return "UP"
    if close < e50 and e50 < e200:
        return "DOWN"
    return "FLAT"


def base_metadata(
    candles: Sequence[Kline],
    *,
    vol_window: int = 20,
    atr_period: int = 14,
) -> Dict[str, Any]:
    closes = [c.close for c in candles]
    vols = [c.volume for c in candles]

    e20_series = ema(closes, 20)
    e50_series = ema(closes, 50)
    e200_series = ema(closes, 200)

    e20 = _last_valid(e20_series)
    e50 = _last_valid(e50_series)
    e200 = _last_valid(e200_series)

    last = candles[-1]
    atr_values = atr(candles, atr_period)
    atr_value = _last_valid(atr_values)
    if math.isnan(atr_value):
        atr_value = 0.0

    atr_pct = (atr_value / last.close) if last.close else 0.0

    vol_slice = vols[-vol_window:] if len(vols) >= vol_window else vols
    avg_vol = fmean(vol_slice) if vol_slice else 0.0
    rel_vol = (last.volume / avg_vol) if avg_vol else 0.0

    trend = trend_label(last.close, e20, e50, e200)

    return {
        "trend": trend,
        "ema_fast": e20,
        "ema_slow": e50,
        "ema_anchor": e200,
        "atr": atr_value,
        "atr_pct": atr_pct,
        "rel_volume": rel_vol,
        "ref_price": last.close,
        "last_close": last.close,
        "last_volume": last.volume,
    }


def passes_sanity(
    meta: Dict[str, Any],
    *,
    min_atr_pct: float = 0.0008,
    min_rel_vol: float = 0.9,
) -> bool:
    atr_pct = meta.get("atr_pct")
    if atr_pct is None or atr_pct < min_atr_pct:
        return False
    rel_vol = meta.get("rel_volume")
    if rel_vol is None or rel_vol < min_rel_vol:
        return False
    return True


__all__ = ["base_metadata", "passes_sanity", "trend_label"]
