from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class RsiPullbackStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="RSI Pullback Trend",
            abbreviation="RSI",
            interval="1m",
            lookback=50,
        )

    def rsi(self, closes: list[float], n: int) -> float | None:
        if len(closes) < n + 1:
            return None
        gains = [max(closes[i] - closes[i-1], 0) for i in range(-n, 0)]
        losses = [max(closes[i-1] - closes[i], 0) for i in range(-n, 0)]
        avg_gain, avg_loss = sum(gains)/n, sum(losses)/n
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 52:
            return []
        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]
        sma_trend = sum(closes[-50:]) / 50
        r = self.rsi(closes, 14)
        v_med = sorted(vols[-50:])[25]
        signals = []
        if last.close >= sma_trend and r is not None and r <= 35 and last.volume >= v_med * 1.2:
            snapshot = {"pattern": "rsi_pullback_long", "rsi": r, "sma_trend": sma_trend}
            signals.append(self.make_signal(symbol, "LONG", confidence=0.75, metadata={"snapshot": snapshot}))
        elif last.close <= sma_trend and r is not None and r >= 65 and last.volume >= v_med * 1.2:
            snapshot = {"pattern": "rsi_pullback_short", "rsi": r, "sma_trend": sma_trend}
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.75, metadata={"snapshot": snapshot}))
        return signals

__all__ = ["RsiPullbackStrategy"]
