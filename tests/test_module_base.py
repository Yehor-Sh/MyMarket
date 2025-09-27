import unittest
from typing import Iterable, List

from binance_client import Kline
from module_base import ModuleBase, Signal


class _DummyClient:
    def __init__(self) -> None:
        self.calls: List[tuple[str, str, int]] = []

    def fetch_klines(self, symbol: str, interval: str, lookback: int) -> List[Kline]:
        self.calls.append((symbol, interval, lookback))
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
            for index in range(lookback)
        ]


class _TestModule(ModuleBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.process_calls: List[tuple[str, List[Kline]]] = []

    def process(self, symbol: str, candles: Iterable[Kline]) -> Iterable[Signal]:
        candles_list = list(candles)
        self.process_calls.append((symbol, candles_list))
        return []


class ModuleBaseTests(unittest.TestCase):
    def test_default_configuration_does_not_request_extra_timeframes(self) -> None:
        client = _DummyClient()
        module = _TestModule(
            client,
            name="Test",
            abbreviation="TST",
            interval="1m",
            lookback=3,
        )

        signals = module.get_signals(["BTCUSDT"])

        self.assertEqual(signals, [])
        self.assertEqual(client.calls, [("BTCUSDT", "1m", 3)])
        self.assertEqual(len(module.process_calls), 1)

    def test_explicit_extra_timeframes_are_respected(self) -> None:
        client = _DummyClient()
        module = _TestModule(
            client,
            name="Test",
            abbreviation="TST",
            interval="1m",
            lookback=3,
            extra_timeframes={"1h": 2},
        )

        signals = module.get_signals(["ETHUSDT"])

        self.assertEqual(signals, [])
        self.assertEqual(
            client.calls,
            [("ETHUSDT", "1m", 3), ("ETHUSDT", "1h", 2)],
        )
        self.assertEqual(len(module.process_calls), 1)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    unittest.main()
