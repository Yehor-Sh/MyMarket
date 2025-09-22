"""Base classes and helpers for strategy modules."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence

from binance_client import BinanceClient, Kline

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
    klines and returns zero or more :class:`Signal` instances.
    """

    def __init__(
        self,
        client: BinanceClient,
        *,
        name: str,
        abbreviation: str,
        interval: str,
        lookback: int,
    ) -> None:
        self.client = client
        self.name = name
        self.abbreviation = abbreviation
        self.interval = interval
        self.lookback = lookback

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
                candles = self.client.fetch_klines(symbol, self.interval, self.lookback)
            except Exception:  # pragma: no cover - defensive
                _logger.exception("failed to fetch klines for %s", symbol)
                continue
            if len(candles) < self.minimum_bars:
                continue
            try:
                signals = list(self.process(symbol, candles))
                results.extend(signals)
            except Exception:  # pragma: no cover - defensive
                _logger.exception("module %s failed for %s", self.name, symbol)
        return results

    # ------------------------------------------------------------------
    @abstractmethod
    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        """Inspect the provided candles and yield signals."""

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
