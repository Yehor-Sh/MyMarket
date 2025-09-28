from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal


class HammerStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Hammer Pattern",
            abbreviation="HAM",
            interval="15m",
            lookback=5,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 3:
            return []

        latest = candles[-1]
        prev = candles[-2]

        body = abs(latest.close - latest.open)
        lower_shadow = min(latest.open, latest.close) - latest.low
        upper_shadow = latest.high - max(latest.open, latest.close)

        if lower_shadow >= 2 * body and upper_shadow <= body * 0.3 and latest.close > prev.close:
            snapshot = {
                "open": latest.open,
                "close": latest.close,
                "high": latest.high,
                "low": latest.low,
                "body": body,
                "lower_shadow": lower_shadow,
                "upper_shadow": upper_shadow,
            }
            signal = self.make_signal(symbol, "LONG", confidence=0.8, metadata=snapshot)
            return [signal]

        return []


__all__ = ["HammerStrategy"]
