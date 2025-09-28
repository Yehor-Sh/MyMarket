from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class DonchianBreakoutStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="Donchian Breakout",
            abbreviation="DCB",
            interval="1m",
            lookback=50,
        )

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles) < 51:
            return []
        window = candles[-51:-1]
        last = candles[-1]
        highs = [c.high for c in window]
        lows = [c.low for c in window]
        ref_high, ref_low = max(highs), min(lows)
        vols = [c.volume for c in window]
        v_med = sorted(vols)[len(vols)//2]
        last_vol = last.volume
        signals = []
        if last.close >= ref_high * 1.0008 and last_vol >= v_med * 1.4:
            snapshot = {"pattern": "donchian_breakout_up", "lookback": 50, "ref_high": ref_high}
            signals.append(self.make_signal(symbol, "LONG", confidence=0.8, metadata={"snapshot": snapshot}))
        elif last.close <= ref_low * (1.0 - 0.0008) and last_vol >= v_med * 1.4:
            snapshot = {"pattern": "donchian_breakout_down", "lookback": 50, "ref_low": ref_low}
            signals.append(self.make_signal(symbol, "SHORT", confidence=0.8, metadata={"snapshot": snapshot}))
        return signals

__all__ = ["DonchianBreakoutStrategy"]
