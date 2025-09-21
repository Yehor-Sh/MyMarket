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


__all__ = ["ema", "rsi", "atr"]
