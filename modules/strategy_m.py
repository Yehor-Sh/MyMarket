from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class ReversalMultiTFStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Reversal Multi-TF",
            abbreviation="RMT",
            interval="15m",
            lookback=30,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 3:
            return []
        current, prev1, prev2 = candles[-1], candles[-2], candles[-3]
        signals = []
        body = abs(current.close - current.open)
        total_range = current.high - current.low
        lower_shadow = min(current.open,current.close)-current.low
        upper_shadow = current.high - max(current.open,current.close)
        # Engulfing
        if prev1.close < prev1.open and current.close > current.open and current.open < prev1.close and current.close > prev1.open:
            signals.append(self.make_signal(symbol,"LONG",confidence=0.8,metadata={"snapshot":{"pattern":"BULL_ENGULF"}}))
        if prev1.close > prev1.open and current.close < current.open and current.open > prev1.close and current.close < prev1.open:
            signals.append(self.make_signal(symbol,"SHORT",confidence=0.8,metadata={"snapshot":{"pattern":"BEAR_ENGULF"}}))
        # Hammer/Hanging man
        if lower_shadow >= 2*body and upper_shadow <= 0.1*total_range and current.close>current.open:
            signals.append(self.make_signal(symbol,"LONG",confidence=0.75,metadata={"snapshot":{"pattern":"HAMMER"}}))
        if lower_shadow >= 2*body and upper_shadow <= 0.1*total_range and current.close<current.open:
            signals.append(self.make_signal(symbol,"SHORT",confidence=0.75,metadata={"snapshot":{"pattern":"HANGING_MAN"}}))
        # Morning star
        if prev2.close<prev2.open and abs(prev1.close-prev1.open)/prev1.open<0.01 and current.close>current.open and current.close>prev2.open*0.5:
            signals.append(self.make_signal(symbol,"LONG",confidence=0.8,metadata={"snapshot":{"pattern":"MORNING_STAR"}}))
        # Evening star
        if prev2.close>prev2.open and abs(prev1.close-prev1.open)/prev1.open<0.01 and current.close<current.open and current.close<prev2.open*1.5:
            signals.append(self.make_signal(symbol,"SHORT",confidence=0.8,metadata={"snapshot":{"pattern":"EVENING_STAR"}}))
        return signals

__all__ = ["ReversalMultiTFStrategy"]
