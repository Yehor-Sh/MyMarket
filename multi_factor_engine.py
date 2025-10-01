"""Multi-factor validation engine for post-processing trading signals."""

from __future__ import annotations

import logging
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence

from binance_client import Kline
from module_base import Signal

FactorCallable = Callable[[Signal, Sequence[Kline] | None, Dict[str, object]], bool]

_logger = logging.getLogger(__name__)


def _factor_name(factor: FactorCallable) -> str:
    name = getattr(factor, "__name__", "")
    if name and name != "<lambda>":
        return name
    qualname = getattr(factor, "__qualname__", "")
    if qualname:
        return qualname
    named = getattr(factor, "name", "")
    if named:
        return str(named)
    return factor.__class__.__name__


def _true_range(previous_close: float, candle: Kline) -> float:
    high_low = candle.high - candle.low
    high_close = abs(candle.high - previous_close)
    low_close = abs(candle.low - previous_close)
    return max(high_low, high_close, low_close)


def calculate_atr(candles: Sequence[Kline], period: int = 14) -> float:
    if period <= 0 or len(candles) <= period:
        return 0.0
    atr_values: List[float] = []
    previous_close = candles[-period - 1].close
    for candle in candles[-period:]:
        atr_values.append(_true_range(previous_close, candle))
        previous_close = candle.close
    if not atr_values:
        return 0.0
    return sum(atr_values) / len(atr_values)


def calculate_rsi(candles: Sequence[Kline], period: int = 14) -> float:
    if period <= 0 or len(candles) <= period:
        return 50.0
    gains: List[float] = []
    losses: List[float] = []
    closes = [candle.close for candle in candles[-(period + 1) :]]
    for prev, curr in zip(closes, closes[1:]):
        change = curr - prev
        if change > 0:
            gains.append(change)
        elif change < 0:
            losses.append(-change)
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def check_trend(signal: Signal, candles: Sequence[Kline] | None, context: Dict[str, object]) -> bool:
    if not candles or len(candles) < 5:
        return False
    closes = [candle.close for candle in candles[-20:]]
    if len(closes) < 2:
        return False
    slope = closes[-1] - closes[0]
    if abs(slope) < 1e-8:
        return False
    return slope > 0 if signal.side == "LONG" else slope < 0


def check_volume(signal: Signal, candles: Sequence[Kline] | None, context: Dict[str, object]) -> bool:
    if not candles or len(candles) < 21:
        return False
    lookback = candles[-21:-1]
    average_volume = mean(candle.volume for candle in lookback) if lookback else 0.0
    last_volume = candles[-1].volume
    if average_volume <= 0:
        return False
    return last_volume >= average_volume * 1.2


def check_atr(signal: Signal, candles: Sequence[Kline] | None, context: Dict[str, object]) -> bool:
    if not candles:
        return False
    atr = calculate_atr(candles, period=min(14, len(candles) - 1))
    if atr <= 0:
        return False
    price = candles[-1].close
    if price <= 0:
        return False
    return atr / price >= 0.003


def check_levels(signal: Signal, candles: Sequence[Kline] | None, context: Dict[str, object]) -> bool:
    if not candles or len(candles) < 10:
        return False
    window = candles[-50:] if len(candles) > 50 else candles
    last_close = window[-1].close
    atr = calculate_atr(window, period=min(14, len(window) - 1))
    if atr <= 0:
        atr = (max(candle.high for candle in window) - min(candle.low for candle in window)) / max(1, len(window))
    tolerance = max(atr * 1.5, last_close * 0.002)
    if signal.side == "LONG":
        support = min(candle.low for candle in window)
        return abs(last_close - support) <= tolerance
    resistance = max(candle.high for candle in window)
    return abs(resistance - last_close) <= tolerance


def check_rsi(signal: Signal, candles: Sequence[Kline] | None, context: Dict[str, object]) -> bool:
    if not candles or len(candles) < 15:
        return False
    rsi = calculate_rsi(candles, period=min(14, len(candles) - 1))
    if signal.side == "LONG":
        return rsi < 70
    return rsi > 30


def check_global(signal: Signal, candles: Sequence[Kline] | None, context: Dict[str, object]) -> bool:
    if not context:
        return True
    trends: List[str] = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        entry = context.get(symbol)
        if isinstance(entry, dict):
            trend = entry.get("trend")
            if isinstance(trend, str):
                trends.append(trend.upper())
    if not trends:
        return True
    if signal.side == "LONG":
        if "DOWN" in trends:
            return False
        return "UP" in trends or "FLAT" in trends
    if "UP" in trends:
        return False
    return "DOWN" in trends or "FLAT" in trends


DEFAULT_FACTORS: List[FactorCallable] = [
    check_trend,
    check_volume,
    check_atr,
    check_levels,
    check_rsi,
    check_global,
]


class MultiFactorEngine:
    """Evaluate cluster signals against a configurable list of factors."""

    def __init__(
        self,
        factors: Iterable[FactorCallable] | None = None,
        *,
        min_pass: int = 2,
    ) -> None:
        self._factors: List[FactorCallable] = [
            factor for factor in (factors or []) if callable(factor)
        ]
        self.min_pass = max(0, int(min_pass))
        if not self._factors:
            _logger.warning(
                "MultiFactorEngine initialised without factors; signals will pass through.",
            )

    @property
    def has_factors(self) -> bool:
        return bool(self._factors)

    def validate(
        self,
        signals: Iterable[Signal],
        candles: Dict[str, Sequence[Kline]] | None,
        context: Dict[str, object] | None,
    ) -> List[Signal]:
        validated: List[Signal] = []
        total = len(self._factors)
        required_default = min(self.min_pass, total) if total else 0
        candle_map = candles or {}
        context_map: Dict[str, object] = context or {}
        for signal in signals:
            symbol = signal.symbol.upper()
            symbol_candles = candle_map.get(symbol, [])
            passed: List[str] = []
            if total:
                for factor in self._factors:
                    name = _factor_name(factor)
                    try:
                        if factor(signal, symbol_candles, context_map):
                            passed.append(name)
                    except Exception:  # pragma: no cover - defensive
                        _logger.exception(
                            "factor %s failed for %s", name, symbol
                        )
            if passed:
                passed = list(dict.fromkeys(passed))
            score = len(passed)
            required = required_default if total else 0
            metadata = dict(signal.metadata)
            metadata.update(
                {
                    "factors_passed": passed,
                    "factors_total": total,
                    "factors_required": required,
                    "factors_score": score,
                    "factors_ratio": (score / total) if total else 1.0,
                }
            )
            if total and score < required:
                continue
            validated.append(
                Signal(
                    symbol=signal.symbol,
                    side=signal.side,
                    strategy=signal.strategy,
                    confidence=signal.confidence,
                    metadata=metadata,
                )
            )
        return validated


__all__ = [
    "FactorCallable",
    "MultiFactorEngine",
    "DEFAULT_FACTORS",
    "calculate_atr",
    "calculate_rsi",
    "check_trend",
    "check_volume",
    "check_atr",
    "check_levels",
    "check_rsi",
    "check_global",
]
