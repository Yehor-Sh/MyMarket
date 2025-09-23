"""Template for creating new strategy modules."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal


class SkeletonStrategy(ModuleBase):
    """Reference implementation for bootstrapping new strategies."""

    def __init__(
        self,
        client: BinanceClient,
        *,
        interval: str = "1h",
        lookback: int = 200,
        extra_timeframes: Mapping[str, int] | None = None,
    ) -> None:
        # TODO: Adjust the human-readable name and abbreviation for the new strategy.
        super().__init__(
            client,
            name="Skeleton Strategy",
            abbreviation="SKEL",
            interval=interval,
            lookback=lookback,
            extra_timeframes=extra_timeframes,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        """Delegate to :meth:`analyze` until custom processing is implemented."""

        # TODO: Replace this simple wrapper if the strategy needs to emit
        # multiple signals or operate on additional timeframes.
        signal = self.analyze(symbol, candles)
        return [signal] if signal else []

    def analyze(self, symbol: str, candles: Sequence[Kline]) -> Signal | None:
        """Evaluate ``candles`` and optionally emit a trading signal."""

        # TODO: Replace placeholder logic with actual indicator calculations.
        # indicators = self._compute_indicators(candles)

        # TODO: Evaluate risk management or confirmation filters.
        # if not self._passes_filters(indicators):
        #     return None

        # TODO: Implement entry/exit decision making and create a signal.
        # if entry_condition:
        #     return self.make_signal(symbol, "LONG", confidence=1.0)

        return None

    def __str__(self) -> str:
        return self.name


__all__ = ["SkeletonStrategy"]
