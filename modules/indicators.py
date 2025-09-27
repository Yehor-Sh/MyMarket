"""Technical indicators and supporting helpers used by strategy modules.

The utilities in this module provide the common building blocks shared by
multiple strategies: raw indicator calculations (EMA, RSI, ATR) together with
metadata helpers that aggregate those indicators into a convenient structure and
lightweight sanity filters for generated signals.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Dict, List, Sequence

from binance_client import Kline


def ema(values: Sequence[float], period: int) -> List[float]:
    """Compute an exponential moving average for ``values``."""

    if period <= 0:
        raise ValueError("period must be positive")

    n = len(values)
    if n == 0:
        return []

    ema_values: List[float] = [math.nan] * n
    if n < period:
        return ema_values

    multiplier = 2.0 / (period + 1)
    sma = sum(values[:period]) / period
    ema_values[period - 1] = sma

    for idx in range(period, n):
        prev = ema_values[idx - 1]
        if math.isnan(prev):
            prev = sma
        current = (values[idx] - prev) * multiplier + prev
        ema_values[idx] = current
    return ema_values


def rsi(values: Sequence[float], period: int = 14) -> List[float]:
    """Compute the relative strength index for ``values``."""

    if period <= 0:
        raise ValueError("period must be positive")

    n = len(values)
    if n == 0:
        return []

    rsi_values: List[float] = [math.nan] * n
    if n <= period:
        return rsi_values

    gains: List[float] = []
    losses: List[float] = []
    for idx in range(1, n):
        change = values[idx] - values[idx - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def _compute(avg_g: float, avg_l: float) -> float:
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return 100.0 - (100.0 / (1.0 + rs))

    rsi_values[period] = _compute(avg_gain, avg_loss)

    for idx in range(period + 1, n):
        gain = gains[idx - 1]
        loss = losses[idx - 1]
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        rsi_values[idx] = _compute(avg_gain, avg_loss)

    return rsi_values


def atr(candles: Sequence[Kline], period: int = 14) -> List[float]:
    """Average true range for the provided ``candles``."""

    if period <= 0:
        raise ValueError("period must be positive")

    n = len(candles)
    if n == 0:
        return []

    atr_values: List[float] = [math.nan] * n
    true_ranges: List[float] = []
    for idx, candle in enumerate(candles):
        if idx == 0:
            true_ranges.append(candle.high - candle.low)
            continue
        prev_close = candles[idx - 1].close
        tr = max(
            candle.high - candle.low,
            abs(candle.high - prev_close),
            abs(candle.low - prev_close),
        )
        true_ranges.append(tr)

    if n < period:
        return atr_values

    initial_atr = sum(true_ranges[:period]) / period
    atr_values[period - 1] = initial_atr

    for idx in range(period, n):
        prev_atr = atr_values[idx - 1]
        if math.isnan(prev_atr):
            prev_atr = initial_atr
        current_atr = ((prev_atr * (period - 1)) + true_ranges[idx]) / period
        atr_values[idx] = current_atr

    return atr_values


def _last_valid(values: Sequence[float]) -> float:
    """Return the last non-NaN value in ``values`` or ``nan`` when missing."""

    for value in reversed(values):
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return float(value)
    return float("nan")


def trend_label(close: float, e20: float, e50: float, e200: float) -> str:
    """Classify the prevailing trend based on key EMA levels."""

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
    """Build strategy metadata derived from ``candles`` and key indicators."""

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
    """Check whether base metadata satisfies minimal signal requirements."""

    atr_pct = meta.get("atr_pct")
    if atr_pct is None or atr_pct < min_atr_pct:
        return False
    rel_vol = meta.get("rel_volume")
    if rel_vol is None or rel_vol < min_rel_vol:
        return False
    return True


__all__ = [
    "ema",
    "rsi",
    "atr",
    "base_metadata",
    "passes_sanity",
    "trend_label",
]
