from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class MomentumScalperStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Momentum Scalper",
            abbreviation="MSC",
            interval="1m",
            lookback=10,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 6:
            return []
        last = candles[-1]
        prev = candles[-2]
        avg_vol = sum(c.volume for c in candles[-10:-5]) / 5
        if last.volume < avg_vol * 1.8:
            return []
        price_change = (last.close - prev.close) / prev.close * 100
        if abs(price_change) < 1.2:
            return []
        # Confirmation
        same_dir = 0
        for i in range(2,4):
            if len(candles) <= i: break
            prev_candle = candles[-i]
            prev_change = (prev_candle.close - candles[-i-1].close)/candles[-i-1].close*100
            if (price_change > 0 and prev_change > 0) or (price_change < 0 and prev_change < 0):
                same_dir += 1
        if same_dir < 1:
            return []
        direction = "LONG" if price_change > 0 else "SHORT"
        snapshot = {"pattern":"momentum_scalper","price_change":price_change,"volume":last.volume,"avg_vol":avg_vol}
        return [self.make_signal(symbol, direction, confidence=0.85, metadata={"snapshot":snapshot})]

__all__ = ["MomentumScalperStrategy"]
