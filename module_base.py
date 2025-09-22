"""Base classes and helpers for strategy modules."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from binance_client import BinanceClient, Kline
from multi_timeframe_config import MULTI_TIMEFRAME_CONFIG

SignalSide = Literal["LONG", "SHORT"]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Signal:
    """Represents a trading signal emitted by a module."""

    symbol: str
    side: SignalSide
    strategy: str
    confidence: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)


class ModuleBase(ABC):
    """Common utilities for strategy modules.

    Sub-classes must implement :meth:`process` that inspects a sequence of
    klines and returns zero or more :class:`Signal` instances.  Multi-timeframe
    strategies can either declare their additional requirements via
    ``multi_timeframe_config`` or override :meth:`process_with_timeframes` to
    handle the extra candle data.
    """

    def __init__(
        self,
        client: BinanceClient,
        *,
        name: str,
        abbreviation: str,
        interval: str,
        lookback: int,
        extra_timeframes: Mapping[str, int] | None = None,
    ) -> None:
        self.client = client
        self.name = name
        self.abbreviation = abbreviation
        self.interval = interval
        self.lookback = lookback
        if extra_timeframes is not None:
            self.extra_timeframes: Dict[str, int] = dict(extra_timeframes)
        else:
            self.extra_timeframes = dict(
                MULTI_TIMEFRAME_CONFIG.get(self.abbreviation, {})
            )

    @property
    def minimum_bars(self) -> int:
        """Minimal number of bars required for analysis."""

        return self.lookback

    # ------------------------------------------------------------------
    def get_signals(self, symbols: Iterable[str]) -> List[Signal]:
        """Fetch klines for ``symbols`` and analyse them."""

        results: List[Signal] = []
        for symbol in symbols:
            try:
                primary_candles = self.client.fetch_klines(
                    symbol, self.interval, self.lookback
                )
            except Exception:  # pragma: no cover - defensive
                _logger.exception("failed to fetch klines for %s", symbol)
                continue
            if len(primary_candles) < self.minimum_bars:
                continue

            extra_candles: Dict[str, Sequence[Kline]] = {}
            missing_timeframe_data = False
            for extra_interval, extra_lookback in self.extra_timeframes.items():
                try:
                    candles = self.client.fetch_klines(
                        symbol, extra_interval, extra_lookback
                    )
                except Exception:  # pragma: no cover - defensive
                    _logger.exception(
                        "failed to fetch %s klines for %s", extra_interval, symbol
                    )
                    missing_timeframe_data = True
                    break
                if len(candles) < extra_lookback:
                    missing_timeframe_data = True
                    break
                extra_candles[extra_interval] = candles

            if missing_timeframe_data:
                continue
            try:
                signals = list(
                    self.process_with_timeframes(
                        symbol, primary_candles, extra_candles
                    )
                )
                results.extend(signals)
            except Exception:  # pragma: no cover - defensive
                _logger.exception("module %s failed for %s", self.name, symbol)
        return results

    # ------------------------------------------------------------------
    @abstractmethod
    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        """Inspect the provided candles and yield signals."""

    # ------------------------------------------------------------------
    def process_with_timeframes(
        self,
        symbol: str,
        primary_candles: Sequence[Kline],
        extra_candles: Mapping[str, Sequence[Kline]],
    ) -> Iterable[Signal]:
        """Inspect candles from the primary and any additional timeframes.

        The default implementation delegates to :meth:`process` for backward
        compatibility so that existing single-timeframe modules continue to
        function without modification.  Multi-timeframe strategies can override
        this hook to consume the ``extra_candles`` mapping.
        """

        return self.process(symbol, primary_candles)

    # ------------------------------------------------------------------
    def run(self, symbol: str, candles: Sequence[Kline]) -> Optional[Signal]:
        """Execute the strategy for a single symbol.

        The default implementation delegates to :meth:`process` and returns the
        first signal emitted, if any.  Strategies are free to override this if
        they prefer a different behaviour.
        """

        signals = list(self.process(symbol, candles))
        return signals[0] if signals else None

    # ------------------------------------------------------------------
    def make_signal(
        self,
        symbol: str,
        side: SignalSide,
        *,
        confidence: float = 1.0,
        metadata: Dict[str, object] | None = None,
    ) -> Signal:
        return Signal(
            symbol=symbol.upper(),
            side=side,
            strategy=self.abbreviation,
            confidence=confidence,
            metadata=metadata or {},
        )


__all__ = ["ModuleBase", "Signal", "SignalSide"]
