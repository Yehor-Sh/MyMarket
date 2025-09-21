import unittest
from datetime import datetime, timezone

from trading_bot.trading_bot import PriceProcessor


class PriceProcessorTestCase(unittest.TestCase):
    def test_trade_ticks_are_aggregated_into_candles(self) -> None:
        processor = PriceProcessor(interval_seconds=60)
        first = {"price": 100.0, "volume": 1.0, "timestamp": 1_700_000_000}
        second = {"price": 101.0, "volume": 0.5, "timestamp": 1_700_000_030}
        third = {"price": 102.0, "volume": 0.7, "timestamp": 1_700_000_060}

        self.assertIsNone(processor.process_tick(first))
        self.assertIsNone(processor.process_tick(second))
        candle = processor.process_tick(third)
        self.assertIsNotNone(candle)
        assert candle is not None
        self.assertAlmostEqual(candle.open, 100.0)
        self.assertAlmostEqual(candle.close, 101.0)
        self.assertAlmostEqual(candle.high, 101.0)
        self.assertAlmostEqual(candle.low, 100.0)
        self.assertAlmostEqual(candle.volume, 1.5)

        trailing = processor.flush()
        self.assertIsNotNone(trailing)
        assert trailing is not None
        self.assertAlmostEqual(trailing.open, 102.0)
        self.assertAlmostEqual(trailing.close, 102.0)

    def test_binance_kline_message(self) -> None:
        processor = PriceProcessor(interval_seconds=60)
        message = {
            "k": {
                "t": 1_700_000_000_000,
                "T": 1_700_000_060_000,
                "o": "100.0",
                "h": "105.0",
                "l": "99.0",
                "c": "104.0",
                "v": "12.5",
                "x": True,
            }
        }
        candle = processor.process_tick(message)
        self.assertIsNotNone(candle)
        assert candle is not None
        self.assertAlmostEqual(candle.open, 100.0)
        self.assertAlmostEqual(candle.high, 105.0)
        self.assertAlmostEqual(candle.low, 99.0)
        self.assertAlmostEqual(candle.close, 104.0)
        self.assertAlmostEqual(candle.volume, 12.5)
        self.assertEqual(candle.open_time.tzinfo, timezone.utc)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
