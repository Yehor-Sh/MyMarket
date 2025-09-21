"""Pin bar reversal strategy."""

from __future__ import annotations

from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class PinBarStrategy(ModuleBase):
    """Identify pin bar candles suggesting possible reversals."""

    def __init__(self, client, *, interval: str = "15m", lookback: int = 50) -> None:
        super().__init__(
            client,
            name="PinBar",
            abbreviation="PIN",
            interval=interval,
            lookback=max(lookback, 20),
        )

    def process(self, symbol: str, candles: List[Kline]) -> Iterable[Signal]:
        curr = candles[-1]
        prev = candles[-2]
        signals: List[Signal] = []

        body = max(curr.body, 1e-8)
        if curr.lower_wick >= body * 2.5 and curr.upper_wick <= body * 0.5 and curr.is_bullish:
            metadata = {
                "body": body,
                "lower_wick": curr.lower_wick,
                "upper_wick": curr.upper_wick,
                "prev_close": prev.close,
            }
            signals.append(self.make_signal(symbol, "LONG", confidence=0.9, metadata=metadata))
        if curr.upper_wick >= body * 2.5 and curr.lower_wick <= body * 0.5 and curr.is_bearish:
            metadata = {
                "body": body,
                "lower_wick": curr.lower_wick,
                "upper_wick": curr.upper_wick,
                "prev_close": prev.close,
            }
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.9, metadata=metadata))
        return signals


__all__ = ["PinBarStrategy"]
