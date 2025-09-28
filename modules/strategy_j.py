from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class MacdSurgeStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="MACD Surge",
            abbreviation="MACD",
            interval="1m",
            lookback=60,
        )

    def ema(self, values: list[float], n: int) -> list[float] | None:
        if len(values) < n:
            return None
        k = 2.0 / (n + 1)
        res = []
        s = sum(values[:n]) / n
        res.extend([None] * (n - 1))
        res.append(s)
        for v in values[n:]:
            s = v * k + s * (1 - k)
            res.append(s)
        return res

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 30:
            return []
        closes = [c.close for c in candles]
        vols = [c.volume for c in candles]
        last = candles[-1]

        # EMA fast/slow
        e_fast = self.ema(closes, 12)
        e_slow = self.ema(closes, 26)
        if e_fast is None or e_slow is None:
            return []

        macd_line = [
            (ef - es) if ef is not None and es is not None else None
            for ef, es in zip(e_fast, e_slow)
        ]
        macd_compact = [x for x in macd_line if x is not None]

        # Signal line
        sig = self.ema(macd_compact, 9)
        if sig is None:
            return []

        # Histogram (игнорируем None)
        hist = [m - s for m, s in zip(macd_compact[-len(sig):], sig) if s is not None]
        if len(hist) < 5:
            return []

        last_hist, prev_hist = hist[-1], hist[-2]
        abs_hist = [abs(x) for x in hist[-min(40, len(hist)):]]
        m_hist = sorted(abs_hist)[len(abs_hist) // 2]
        v_med = sorted(vols[-30:])[15]

        signals = []
        if last_hist > 0 and last_hist >= m_hist * 1.5 and prev_hist <= 0 and last.volume >= v_med * 1.3:
            snapshot = {"pattern": "macd_surge_up", "hist": last_hist}
            signals.append(self.make_signal(symbol, "LONG", confidence=0.8, metadata={"snapshot": snapshot}))
        elif last_hist < 0 and abs(last_hist) >= m_hist * 1.5 and prev_hist >= 0 and last.volume >= v_med * 1.3:
            snapshot = {"pattern": "macd_surge_down", "hist": last_hist}
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.8, metadata={"snapshot": snapshot}))
        return signals

__all__ = ["MacdSurgeStrategy"]
