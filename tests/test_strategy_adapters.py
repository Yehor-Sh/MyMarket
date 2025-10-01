from __future__ import annotations

from typing import Dict, List

import pytest

from binance_client import Kline


class _FakeClient:
    def __init__(self) -> None:
        self.calls: List[tuple[str, str, int]] = []

    def fetch_klines(self, symbol: str, interval: str, lookback: int) -> List[Kline]:
        self.calls.append((symbol, interval, lookback))
        count = lookback + 5
        return [
            Kline(
                open_time=index,
                open=1.0,
                high=1.5,
                low=0.5,
                close=1.2,
                volume=10.0,
                close_time=index + 1,
            )
            for index in range(count)
        ]


STRATEGY_CONFIG = [
    ("modules.strategy_volume_surge", "VolumeSurgeStrategy", "modules.strategy_volume_surge", "detect_signals"),
    ("modules.strategy_breakout_high", "BreakoutHighStrategy", "modules.strategy_breakout_high", "detect_signals"),
    ("modules.strategy_breakdown_low", "BreakdownLowStrategy", "modules.module_C", "detect_signals"),
    ("modules.strategy_range_surge", "RangeSurgeStrategy", "modules.module_D", "detect_signals"),
    ("modules.strategy_momentum_roc", "MomentumRocStrategy", "modules.module_E", "detect_signals"),
    ("modules.strategy_donchian_breakout", "DonchianBreakoutStrategy", "modules.module_F", "detect"),
    ("modules.strategy_ma_cross_trend", "MaCrossTrendStrategy", "modules.module_G", "detect"),
    ("modules.strategy_rsi_pullback", "RsiPullbackStrategy", "modules.module_H", "detect"),
    ("modules.strategy_bollinger_squeeze", "BollingerSqueezeStrategy", "modules.module_I", "detect"),
    ("modules.strategy_macd_surge", "MacdSurgeStrategy", "modules.module_J", "detect"),
    ("modules.strategy_vwap_reversion", "VwapReversionStrategy", "modules.module_K", "detect"),
]


@pytest.mark.parametrize("strategy_module, class_name, detect_module, detect_attr", STRATEGY_CONFIG)
def test_strategy_adapter_bridge(monkeypatch, strategy_module: str, class_name: str, detect_module: str, detect_attr: str) -> None:
    module = pytest.importorskip(strategy_module)
    strategy_cls = getattr(module, class_name)

    detect_mod = pytest.importorskip(detect_module)

    calls: Dict[str, int] = {}

    def _fake_detect(snapshot: Dict[str, object], state: Dict[str, object]):
        symbols = snapshot.get("symbols")
        assert isinstance(symbols, dict) and symbols
        symbol = next(iter(symbols))
        calls[symbol] = calls.get(symbol, 0) + 1
        state_key = f"counter:{symbol}"
        state[state_key] = state.get(state_key, 0) + 1
        return [
            {
                "symbol": symbol,
                "direction": "long",
                "price": 101.0,
                "meta": {"seen": state[state_key]},
            }
        ]

    monkeypatch.setattr(detect_mod, detect_attr, _fake_detect)

    client = _FakeClient()
    strategy = strategy_cls(client)

    first = strategy.get_signals(["BTCUSDT"])
    assert len(first) == 1
    signal = first[0]
    assert signal.symbol == "BTCUSDT"
    assert signal.side == "LONG"
    assert signal.metadata["seen"] == 1
    assert signal.metadata["price"] == 101.0
    assert client.calls[0] == ("BTCUSDT", strategy.interval, strategy.lookback)

    second = strategy.get_signals(["BTCUSDT"])
    assert len(second) == 1
    assert second[0].metadata["seen"] == 2
    assert calls["BTCUSDT"] == 2
