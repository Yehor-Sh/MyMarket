"""Template strategy module with a step-by-step creation checklist.

Checklist for adding a new strategy
===================================
1. **Choose identifiers.** Decide on the human friendly ``name`` and short
   ``abbreviation`` (used as the strategy code inside signals).
2. **Configure market data.** Specify the main ``interval`` and ``lookback``
   window.  If additional timeframes are required, provide them via the
   ``extra_timeframes`` argument or populate
   :data:`multi_timeframe_config.MULTI_TIMEFRAME_CONFIG`.
3. **Initialise ``ModuleBase``.** Forward the Binance client instance together
   with the identifiers and timeframe settings using ``super().__init__``.
4. **Inspect candles in :meth:`process`.** Use the provided sequence of
   :class:`binance_client.Kline` objects (the base class already fetched them)
   to compute indicators, thresholds or other filters.
5. **Prepare a metadata snapshot.** Capture the values that justify a signal in
   a serialisable structure (dict/list/tuple) so UI and tests can display the
   reasoning.
6. **Emit :class:`module_base.Signal` objects.** Use
   :meth:`module_base.ModuleBase.make_signal` to create signals, set ``side``
   (``"LONG"``/``"SHORT"``) and optionally ``confidence``.
7. **Integrate the module.** Add the strategy to worker/orchestrator configs and
   document the new abbreviation so operators can enable it.

The :class:`TemplateStrategy` below keeps the implementation deliberately short
while demonstrating the recommended flow: consume candles, create a snapshot
with the latest values and emit zero or more signals based on simple logic.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal


class TemplateStrategy(ModuleBase):
    """A minimal example strategy that follows the recommended workflow."""

    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Template Strategy",
            abbreviation="TMP",
            interval="1h",
            lookback=20,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        """Generate a bullish signal when the latest candle closes higher."""

        if not candles:
            return []

        latest = candles[-1]
        previous = candles[-2] if len(candles) > 1 else latest

        snapshot = {
            "open_time": latest.open_time,
            "close_time": latest.close_time,
            "open": latest.open,
            "close": latest.close,
            "previous_close": previous.close,
        }

        if latest.close <= latest.open:
            return []

        signal = self.make_signal(
            symbol,
            "LONG",
            confidence=0.5 if latest.close <= previous.close else 0.8,
            metadata={"snapshot": snapshot},
        )
        return [signal]


__all__ = ["TemplateStrategy"]
