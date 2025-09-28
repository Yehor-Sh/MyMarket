from __future__ import annotations
from typing import Iterable, Sequence
from binance_client import BinanceClient
from module_base import Kline, ModuleBase, Signal

class AtrSweepStrategy(ModuleBase):
    def __init__(self, client: BinanceClient) -> None:
        super().__init__(
            client,
            name="ATR Sweep",
            abbreviation="ATS",
            interval="15m",
            lookback=200,
        )

    def atr(self, candles: Sequence[Kline], period: int=14) -> float:
        trs = []
        for i in range(1,len(candles)):
            h,l,c_prev = candles[i].high,candles[i].low,candles[i-1].close
            tr = max(h-l,abs(h-c_prev),abs(l-c_prev))
            trs.append(tr)
        return sum(trs[-period:])/period if len(trs)>=period else 0

    def rsi(self, closes: list[float], period: int=14) -> float:
        delta = [closes[i]-closes[i-1] for i in range(1,len(closes))]
        gains = [d for d in delta if d>0]
        losses = [-d for d in delta if d<0]
        avg_gain = sum(gains[-period:])/period if len(gains)>=period else 0.1
        avg_loss = sum(losses[-period:])/period if len(losses)>=period else 0.1
        rs = avg_gain/avg_loss if avg_loss!=0 else 0
        return 100 - 100/(1+rs)

    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        if len(candles)<200:
            return []
        closes = [c.close for c in candles]
        rsi_val = self.rsi(closes,14)
        atr_val = self.atr(candles,14)
        last = candles[-1]
        prev = candles[-2]
        signals=[]
        if last.low<min(c.low for c in candles[-10:-1]) and last.close>last.open and rsi_val>30 and atr_val/last.close>0.001:
            snapshot={"pattern":"SWEEP_RECLAIM_LONG","rsi":rsi_val,"atr_pct":atr_val/last.close}
            signals.append(self.make_signal(symbol,"LONG",confidence=0.85,metadata={"snapshot":snapshot}))
        if last.high>max(c.high for c in candles[-10:-1]) and last.close<last.open and rsi_val<70 and atr_val/last.close>0.001:
            snapshot={"pattern":"SWEEP_RECLAIM_SHORT","rsi":rsi_val,"atr_pct":atr_val/last.close}
            signals.append(self.make_signal(symbol,"SHORT",confidence=0.85,metadata={"snapshot":snapshot}))
        return signals

__all__ = ["AtrSweepStrategy"]
