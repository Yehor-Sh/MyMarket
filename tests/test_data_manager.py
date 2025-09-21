import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_bot.trading_bot import Candle, DataManager, Trade

UTC = timezone.utc


class DataManagerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        self.candles_path = base / "candles.json"
        self.trades_path = base / "trades.json"
        self.manager = DataManager(self.candles_path, self.trades_path, max_candles=10)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_append_candle_persists_to_disk(self) -> None:
        candle = Candle(
            open_time=datetime.now(tz=UTC),
            close_time=datetime.now(tz=UTC) + timedelta(minutes=1),
            open=100.0,
            high=102.0,
            low=99.5,
            close=101.5,
            volume=12.5,
        )
        self.manager.append_candle(candle)
        payload = self.manager.get_candles_payload()
        self.assertEqual(len(payload["candles"]), 1)
        self.assertAlmostEqual(payload["candles"][0]["close"], 101.5)

        disk_payload = json.loads(self.candles_path.read_text(encoding="utf-8"))
        self.assertEqual(len(disk_payload["candles"]), 1)

    def test_trade_lifecycle(self) -> None:
        trade = Trade(id="trade-1", symbol="TEST", direction="long", entry_price=100.0, stop_loss=95.0)
        self.manager.add_trade(trade)

        trades_payload = self.manager.get_trades_payload()
        self.assertEqual(len(trades_payload["active"]), 1)

        trade.stop_loss = 96.0
        self.manager.update_trade(trade)
        updated_payload = self.manager.get_trades_payload()
        self.assertAlmostEqual(updated_payload["active"][0]["stop_loss"], 96.0)

        now = datetime.now(tz=UTC)
        closed = self.manager.close_trade(trade.id, 110.0, now)
        self.assertIsNotNone(closed)
        final_payload = self.manager.get_trades_payload()
        self.assertEqual(len(final_payload["active"]), 0)
        self.assertEqual(len(final_payload["closed"]), 1)
        self.assertAlmostEqual(final_payload["closed"][0]["exit_price"], 110.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
