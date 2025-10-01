from typing import List

import pytest

from binance_client import BinanceClient, Kline
from module_base import ModuleBase


def _make_kline(open_time: int, close: float) -> Kline:
    return Kline(
        open_time=open_time,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        close_time=open_time + 60_000,
    )


def test_websocket_kline_updates_cache_and_price() -> None:
    client = BinanceClient()
    symbol = "BTCUSDT"
    interval = "1m"
    base_time = 1_000_000
    initial: List[Kline] = [
        _make_kline(base_time, 100.0),
        _make_kline(base_time + 60_000, 101.0),
    ]
    client.prime_klines(symbol, interval, initial)

    payload = {
        "e": "kline",
        "s": symbol,
        "k": {
            "t": base_time + 60_000,
            "T": base_time + 120_000,
            "s": symbol,
            "i": interval,
            "o": "101.0",
            "c": "105.0",
            "h": "106.5",
            "l": "99.5",
            "v": "12.0",
            "x": False,
        },
    }

    client._process_ws_payload(payload)

    cached = client.get_cached_klines(symbol, interval)
    assert len(cached) == 2
    forming = cached[-1]
    assert forming.close == pytest.approx(105.0)
    assert forming.high == pytest.approx(106.5)
    assert forming.low == pytest.approx(99.5)
    assert forming.volume == pytest.approx(12.0)

    payload["k"].update({"c": "108.0", "h": "110.0", "l": "100.0", "v": "20.0", "x": True})
    client._process_ws_payload(payload)

    final = client.get_cached_klines(symbol, interval)[-1]
    assert final.close == pytest.approx(108.0)
    assert final.high == pytest.approx(110.0)
    assert final.low == pytest.approx(100.0)
    assert final.volume == pytest.approx(20.0)
    assert client.get_price(symbol) == pytest.approx(108.0)


def test_module_reads_forming_candle_from_cache() -> None:
    class _SpyModule(ModuleBase):
        def __init__(self, client: BinanceClient) -> None:
            super().__init__(
                client,
                name="Spy",
                abbreviation="SPY",
                interval="1m",
                lookback=2,
            )
            self.last_close: float | None = None

        def process(self, symbol: str, candles: List[Kline]):
            self.last_close = candles[-1].close
            return []

    client = BinanceClient()
    symbol = "ETHUSDT"
    interval = "1m"
    base_time = 2_000_000
    client.prime_klines(
        symbol,
        interval,
        [
            _make_kline(base_time, 50.0),
            _make_kline(base_time + 60_000, 51.0),
        ],
    )

    forming_payload = {
        "e": "kline",
        "s": symbol,
        "k": {
            "t": base_time + 60_000,
            "T": base_time + 120_000,
            "s": symbol,
            "i": interval,
            "o": "51.0",
            "c": "53.0",
            "h": "54.0",
            "l": "50.5",
            "v": "5.0",
            "x": False,
        },
    }
    client._process_ws_payload(forming_payload)

    module = _SpyModule(client)
    module.get_signals([symbol])
    assert module.last_close == pytest.approx(53.0)

    closing_payload = dict(forming_payload)
    closing_payload["k"] = dict(forming_payload["k"], c="55.0", h="56.0", l="51.0", v="8.0", x=True)
    client._process_ws_payload(closing_payload)

    module.get_signals([symbol])
    assert module.last_close == pytest.approx(55.0)
