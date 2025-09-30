from __future__ import annotations

import time
from typing import List, Tuple

import pytest

from binance_client import Kline
from orchestrator import Orchestrator


def _make_candles(prices: List[float]) -> List[Kline]:
    candles: List[Kline] = []
    base_time = 1_000_000
    for index, price in enumerate(prices):
        candles.append(
            Kline(
                open_time=base_time + index * 60_000,
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1_000.0,
                close_time=base_time + (index + 1) * 60_000,
            )
        )
    return candles


def _prepare_orchestrator(
    monkeypatch: pytest.MonkeyPatch, symbol: str, candles: List[Kline]
) -> Tuple[Orchestrator, List[Tuple[str, str, int]]]:
    orchestrator = Orchestrator(cluster_threshold=1)

    # Avoid network requests during tests.
    monkeypatch.setattr(orchestrator, "_broadcast_state", lambda: {})
    monkeypatch.setattr(orchestrator.socketio, "emit", lambda *args, **kwargs: None)

    orchestrator.client.prime_klines(symbol, "4h", candles)
    with orchestrator.client._liquidity_lock:  # type: ignore[attr-defined]
        orchestrator.client._liquid_pairs = [symbol]  # type: ignore[attr-defined]
        orchestrator.client._last_liquidity_refresh = time.time()  # type: ignore[attr-defined]

    fetch_calls: List[Tuple[str, str, int]] = []

    def _fake_fetch(symbol_arg: str, interval: str, limit: int = 100) -> List[Kline]:
        fetch_calls.append((symbol_arg.upper(), interval, limit))
        cached = orchestrator.client.get_cached_klines(symbol_arg, interval)
        if limit and len(cached) > limit:
            return cached[-limit:]
        return cached

    monkeypatch.setattr(orchestrator.client, "fetch_klines", _fake_fetch)

    return orchestrator, fetch_calls


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_handle_signal_uses_primary_interval_cache(monkeypatch: pytest.MonkeyPatch, side: str) -> None:
    symbol = "SOLUSDT" if side == "LONG" else "ADAUSDT"
    prices = [95.0, 96.5, 97.8, 101.2] if side == "LONG" else [101.0, 98.5, 94.3, 90.4]
    candles = _make_candles(prices)

    orchestrator, fetch_calls = _prepare_orchestrator(monkeypatch, symbol, candles)

    strategy = orchestrator.strategies["VWTC4H"]
    signal = strategy.make_signal(symbol, side)

    clustered = orchestrator.cluster_engine.process_signals([signal])
    assert clustered, "clustered signal should be produced when threshold is 1"
    cluster_signal = clustered[0]

    assert orchestrator.client.get_price(symbol) is None
    assert orchestrator.client.get_cached_klines(symbol, "1m") == []

    orchestrator._handle_signal(cluster_signal)

    # 1m fallback should have been attempted and subsequently failed.
    assert (symbol, "1m", 1) in fetch_calls

    cached_after = orchestrator.client.get_cached_klines(symbol, strategy.interval)
    assert len(cached_after) == len(candles)

    assert orchestrator.active_trades, "trade should have been created"
    trade = next(iter(orchestrator.active_trades.values()))
    assert trade.side == side
    assert trade.strategy == cluster_signal.strategy
    assert trade.metadata["cluster_size"] == 1
    assert trade.metadata["strategies"] == [strategy.abbreviation]
    assert trade.entry_price == pytest.approx(candles[-1].close)

