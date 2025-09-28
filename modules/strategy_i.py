from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class BollingerSqueezeBreakoutStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Bollinger Squeeze Breakout",
            abbreviation="BSB",
            interval="1m",
            lookback=60,
        )

    def sma(self, a: list[float], n: int) -> float | None:
        return sum(a[-n:]) / n if len(a) >= n else None

    def std(self, a: list[float], n: int) -> float | None:
        if len(a) < n:
            return None
        s = a[-n:]
        m = sum(s) / n
        v = sum((x - m)**2 for x in s) / n
        return v**0.5

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 22:
            return []
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]
        m = self.sma(closes, 20)
        s = self.std(closes, 20)
        if m is None or s is None:
            return []
        upper, lower = m + 2*s, m - 2*s
        width = upper - lower
        widths = []
        for i in range(20, len(closes)):
            mm = sum(closes[i-20:i]) / 20
            ss = (sum((x-mm)**2 for x in closes[i-20:i]) / 20)**0.5
            widths.append((mm+2*ss) - (mm-2*ss))
        thr = sorted(widths)[max(0, int(len(widths)*0.2)-1)]
        v_med = sorted(vols[-20:])[10]
        last_range = last.high - last.low
        ranges = [highs[i] - lows[i] for i in range(len(highs)-20, len(highs))]
        r_med = sorted(ranges)[10]
        signals = []
        if width <= thr and last.volume >= v_med * 1.4 and last_range >= r_med * 1.2:
            if last.close > upper:
                snapshot = {"pattern": "bollinger_squeeze_breakout_up", "width": width}
                signals.append(self.make_signal(symbol, "LONG", confidence=0.8, metadata={"snapshot": snapshot}))
            elif last.close < lower:
                snapshot = {"pattern": "bollinger_squeeze_breakout_down", "width": width}
                signals.append(self.make_signal(symbol, "SHORT", confidence=0.8, metadata={"snapshot": snapshot}))
        return signals

__all__ = ["BollingerSqueezeBreakoutStrategy"]
