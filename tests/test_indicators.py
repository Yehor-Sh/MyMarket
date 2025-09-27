import math
import unittest

from binance_client import Kline
from modules import indicators


def _make_kline(index: int, close: float, volume: float) -> Kline:
    return Kline(
        open_time=index,
        open=close - 0.2,
        high=close + 0.8,
        low=close - 1.2,
        close=close,
        volume=volume,
        close_time=index,
    )


class TrendLabelTests(unittest.TestCase):
    def test_uptrend_when_close_above_slow_and_anchor(self) -> None:
        self.assertEqual(indicators.trend_label(110.0, 108.0, 105.0, 90.0), "UP")

    def test_downtrend_when_close_below_slow_and_anchor(self) -> None:
        self.assertEqual(indicators.trend_label(80.0, 82.0, 85.0, 95.0), "DOWN")

    def test_flat_when_values_missing(self) -> None:
        self.assertEqual(indicators.trend_label(100.0, math.nan, 95.0, 90.0), "FLAT")


class BaseMetadataTests(unittest.TestCase):
    def test_builds_expected_metadata_fields(self) -> None:
        candles = [
            _make_kline(idx, 100.0 + idx * 0.5, 500.0 + idx * 5.0) for idx in range(250)
        ]
        meta = indicators.base_metadata(candles)

        self.assertEqual(meta["trend"], "UP")
        self.assertGreater(meta["ema_fast"], meta["ema_slow"])
        self.assertGreater(meta["ema_slow"], meta["ema_anchor"])
        self.assertGreater(meta["atr"], 0.0)
        self.assertGreater(meta["atr_pct"], 0.0)
        self.assertGreater(meta["rel_volume"], 0.0)
        self.assertEqual(meta["last_close"], candles[-1].close)
        self.assertEqual(meta["last_volume"], candles[-1].volume)
        self.assertEqual(meta["ref_price"], candles[-1].close)

    def test_passes_sanity_with_default_thresholds(self) -> None:
        candles = [
            _make_kline(idx, 100.0 + idx * 0.5, 500.0 + idx * 5.0) for idx in range(250)
        ]
        meta = indicators.base_metadata(candles)

        self.assertTrue(indicators.passes_sanity(meta))

    def test_passes_sanity_rejects_low_atr_or_volume(self) -> None:
        self.assertFalse(
            indicators.passes_sanity({"atr_pct": 0.0001, "rel_volume": 1.5})
        )
        self.assertFalse(
            indicators.passes_sanity({"atr_pct": 0.0015, "rel_volume": 0.2})
        )


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    unittest.main()
