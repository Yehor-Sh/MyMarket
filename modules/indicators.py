"""Utility functions for common technical indicators used by strategy modules."""

from __future__ import annotations

import math
from typing import List, Sequence

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


def sma(values: Sequence[float], period: int) -> List[float]:
    """Simple moving average for ``values``."""

    if period <= 0:
        raise ValueError("period must be positive")

    n = len(values)
    if n == 0:
        return []

    sma_values: List[float] = [math.nan] * n
    if n < period:
        return sma_values

    window_sum = sum(values[:period])
    sma_values[period - 1] = window_sum / period

    for idx in range(period, n):
        window_sum += values[idx] - values[idx - period]
        sma_values[idx] = window_sum / period

    return sma_values


def rolling_std(values: Sequence[float], period: int) -> List[float]:
    """Rolling standard deviation for ``values`` using a fixed ``period``."""

    if period <= 0:
        raise ValueError("period must be positive")

    n = len(values)
    if n == 0:
        return []

    std_values: List[float] = [math.nan] * n
    if n < period:
        return std_values

    window_sum = sum(values[:period])
    window_sq_sum = sum(value * value for value in values[:period])

    for idx in range(period - 1, n):
        mean = window_sum / period
        variance = (window_sq_sum / period) - (mean * mean)
        variance = max(0.0, variance)
        std_values[idx] = math.sqrt(variance)

        if idx + 1 < n:
            outgoing = values[idx + 1 - period]
            incoming = values[idx + 1]
            window_sum += incoming - outgoing
            window_sq_sum += (incoming * incoming) - (outgoing * outgoing)

    return std_values


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


def bollinger_bands(
    values: Sequence[float],
    period: int = 20,
    *,
    stddev_multiplier: float = 2.0,
) -> tuple[List[float], List[float], List[float], List[float]]:
    """Return middle, upper, lower bands and band width for ``values``."""

    if period <= 0:
        raise ValueError("period must be positive")

    n = len(values)
    if n == 0:
        empty: List[float] = []
        return empty, empty, empty, empty

    middle = sma(values, period)
    deviations = rolling_std(values, period)
    upper: List[float] = [math.nan] * n
    lower: List[float] = [math.nan] * n
    width: List[float] = [math.nan] * n

    for idx in range(n):
        mid = middle[idx]
        dev = deviations[idx]
        if math.isnan(mid) or math.isnan(dev):
            continue
        upper[idx] = mid + (dev * stddev_multiplier)
        lower[idx] = mid - (dev * stddev_multiplier)
        width[idx] = upper[idx] - lower[idx]

    return middle, upper, lower, width


def vwap(candles: Sequence[Kline]) -> List[float]:
    """Volume weighted average price for a sequence of ``candles``."""

    cumulative_tpv = 0.0
    cumulative_volume = 0.0
    result: List[float] = []

    for candle in candles:
        typical_price = (candle.high + candle.low + candle.close) / 3.0
        cumulative_tpv += typical_price * candle.volume
        cumulative_volume += candle.volume
        if cumulative_volume <= 0:
            result.append(math.nan)
        else:
            result.append(cumulative_tpv / cumulative_volume)

    return result


__all__ = [
    "ema",
    "sma",
    "rolling_std",
    "rsi",
    "atr",
    "bollinger_bands",
    "vwap",
]
