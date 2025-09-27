import threading
import unittest
from typing import Iterable, List, Optional

import requests

from binance_client import BinanceClient


class _DummyResponse:
    def __init__(self, status_code: int, payload: Optional[dict] = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def json(self) -> dict:
        return self._payload


class _DummySession:
    def __init__(self, responses: Iterable[_DummyResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: List[dict] = []

    def get(self, url: str, *, params=None, timeout: float) -> _DummyResponse:  # type: ignore[override]
        if self._index >= len(self._responses):
            raise AssertionError("No more responses configured")
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        response = self._responses[self._index]
        self._index += 1
        return response


class _SingleRunEvent:
    """Event-like object that stops a loop after a single iteration."""

    def __init__(self) -> None:
        self._flag = False

    def is_set(self) -> bool:
        return self._flag

    def wait(self, timeout: float) -> bool:
        self._flag = True
        return True

    def set(self) -> None:
        self._flag = True

    def clear(self) -> None:
        self._flag = False


class FetchTickerPricesTests(unittest.TestCase):
    def test_fallback_skips_invalid_symbols(self) -> None:
        session = _DummySession(
            [
                _DummyResponse(400),
                _DummyResponse(400),
                _DummyResponse(200, {"symbol": "ETHUSDT", "price": "3500"}),
            ]
        )
        client = BinanceClient(session=session)

        with self.assertLogs("binance_client", level="WARNING") as captured:
            result = client._fetch_ticker_prices(["INVALIDUSDT", "ETHUSDT"])

        self.assertEqual(result, {"ETHUSDT": 3500.0})
        self.assertTrue(any("INVALIDUSDT" in message for message in captured.output))

    def test_ticker_loop_continues_after_invalid_symbol(self) -> None:
        session = _DummySession(
            [
                _DummyResponse(400),
                _DummyResponse(200, {"symbol": "ETHUSDT", "price": "3500"}),
                _DummyResponse(400),
            ]
        )
        client = BinanceClient(session=session)
        client._subscribed_symbols = {"INVALIDUSDT", "ETHUSDT"}
        client._stop_event = _SingleRunEvent()  # type: ignore[assignment]

        received = []

        def _capture(symbol: str, price: float) -> None:
            received.append((symbol, price))

        client.add_price_listener(_capture)

        thread = threading.Thread(target=client._ticker_loop, daemon=True)
        thread.start()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive(), "ticker loop did not finish in time")
        self.assertEqual(received, [("ETHUSDT", 3500.0)])


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    unittest.main()
