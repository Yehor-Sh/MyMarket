"""Helpers to adapt legacy snapshot-based modules to ``ModuleBase``."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from binance_client import Kline
from module_base import ModuleBase, Signal


class LegacySnapshotStrategy(ModuleBase):
    """Bridge legacy ``detect_signals`` implementations with :class:`ModuleBase`.

    The original standalone scripts consumed full market snapshots represented as
    dictionaries with the structure ``{"symbols": {symbol: [bars...]}}`` where
    each bar was a mapping describing OHLCV values.  The orchestrator, however,
    expects strategy modules to derive from :class:`ModuleBase` and analyse
    :class:`~binance_client.Kline` sequences directly.  This adapter keeps the
    legacy ``detect_signals`` functions untouched while providing the glue code
    required by the worker infrastructure.
    """

    def __init__(
        self,
        client,
        *,
        name: str,
        abbreviation: str,
        interval: str,
        lookback: int,
        detect_func,
    ) -> None:
        super().__init__(
            client,
            name=name,
            abbreviation=abbreviation,
            interval=interval,
            lookback=lookback,
        )
        self._detect_func = detect_func
        self._state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _serialise_candles(candles: Sequence[Kline]) -> List[Dict[str, Any]]:
        """Convert :class:`Kline` objects into the legacy dictionary format."""

        bars: List[Dict[str, Any]] = []
        for candle in candles:
            bars.append(
                {
                    "open_time": candle.open_time,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "close_time": candle.close_time,
                    "is_closed": True,
                }
            )
        return bars

    # ------------------------------------------------------------------
    def _build_snapshot(self, symbol: str, candles: Sequence[Kline]) -> Dict[str, Any]:
        return {
            "symbols": {symbol: self._serialise_candles(candles)},
            "meta": {},
        }

    # ------------------------------------------------------------------
    def _convert_signal(self, raw: Dict[str, Any], symbol: str) -> Signal | None:
        raw_symbol = str(raw.get("symbol", "")).upper()
        if raw_symbol and raw_symbol != symbol.upper():
            return None

        direction = str(raw.get("direction", "")).lower()
        if direction not in ("long", "short"):
            return None

        confidence = float(raw.get("confidence", 1.0))
        metadata: Dict[str, Any] = {}

        if "meta" in raw and isinstance(raw["meta"], dict):
            metadata.update(raw["meta"])

        for key in ("action", "price", "quantity", "stop_loss", "trailing", "time"):
            if key in raw:
                metadata[key] = raw[key]

        side = "LONG" if direction == "long" else "SHORT"
        return self.make_signal(symbol, side, confidence=confidence, metadata=metadata)

    # ------------------------------------------------------------------
    def process(self, symbol: str, candles: Sequence[Kline]) -> Iterable[Signal]:
        snapshot = self._build_snapshot(symbol, candles)
        raw_signals = self._detect_func(snapshot, self._state) or []
        for raw in raw_signals:
            signal = self._convert_signal(raw, symbol)
            if signal is not None:
                yield signal


__all__ = ["LegacySnapshotStrategy"]
