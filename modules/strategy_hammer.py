"""Hammer / shooting star strategy."""

from __future__ import annotations

from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class HammerStrategy(ModuleBase):
    """Detect hammer and inverted hammer candles."""

    def __init__(self, client, *, interval: str = "15m", lookback: int = 50) -> None:
        super().__init__(
            client,
            name="Hammer",
            abbreviation="HAM",
            interval=interval,
            lookback=max(lookback, 20),
        )

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        curr = candles[-1]
        signals: List[Signal] = []

        body = max(curr.body, 1e-8)
        if curr.lower_wick >= body * 2.0 and curr.upper_wick <= body * 0.5:
            metadata = {
                "body": body,
                "lower_wick": curr.lower_wick,
                "upper_wick": curr.upper_wick,
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=0.8, metadata=metadata))
        if curr.upper_wick >= body * 2.0 and curr.lower_wick <= body * 0.5:
            metadata = {
                "body": body,
                "lower_wick": curr.lower_wick,
                "upper_wick": curr.upper_wick,
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.8, metadata=metadata))
        return signals


__all__ = ["HammerStrategy"]
