import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Dict, List

from trading_bot.config import AppConfig, PathsConfig, StrategyConfig, WebSocketConfig
from trading_bot.trading_bot import TradingBot

UTC = timezone.utc


def make_kline(start: int, open_price: float, close_price: float, high: float, low: float, volume: float) -> Dict[str, Dict[str, object]]:
    return {
        "k": {
            "t": start * 1000,
            "T": (start + 60) * 1000,
            "o": str(open_price),
            "h": str(high),
            "l": str(low),
            "c": str(close_price),
            "v": str(volume),
            "x": True,
        }
    }


class FakeFeed:
    def __init__(self, messages: List[Dict[str, object]]) -> None:
        self._messages = list(messages)

    def __aiter__(self) -> AsyncIterator[Dict[str, object]]:
        self._iter = iter(self._messages)
        return self

    async def __anext__(self) -> Dict[str, object]:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class TradingBotIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        paths = PathsConfig(data_dir=base, candles_file=base / "candles.json", trades_file=base / "trades.json")
        websocket = WebSocketConfig(stream_url="ws://example", price_interval_seconds=60, symbol="TESTUSDT")
        strategy = StrategyConfig(lookback=3, fast_ma=2, slow_ma=3, trailing_stop_pct=0.05, min_volume=0.0)
        self.config = AppConfig(websocket=websocket, strategy=strategy, paths=paths, bootstrap_url=None)

    async def asyncTearDown(self) -> None:
        self.tmpdir.cleanup()

    async def test_bot_processes_feed_and_generates_trade(self) -> None:
        feed = FakeFeed(
            [
                make_kline(1_700_000_000, 100.0, 101.0, 101.5, 99.5, 1.0),
                make_kline(1_700_000_060, 101.0, 102.0, 102.5, 100.5, 1.0),
                make_kline(1_700_000_120, 102.0, 104.0, 104.5, 101.5, 1.2),
            ]
        )
        bot = TradingBot(self.config, message_source=feed)
        signals: List[Dict[str, object]] = []
        trades: List[Dict[str, object]] = []
        bot.register_listener("signal", signals.append)
        bot.register_listener("trade", trades.append)

        await bot.run()

        candles_payload = bot.get_candles()
        self.assertEqual(len(candles_payload["candles"]), 3)
        trades_payload = bot.get_trades()
        self.assertEqual(len(trades_payload["active"]), 1)
        self.assertTrue(signals)
        self.assertTrue(trades)

        # Ensure data persisted to disk
        stored_candles = Path(self.config.paths.candles_file).read_text(encoding="utf-8")
        self.assertIn("104.0", stored_candles)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
