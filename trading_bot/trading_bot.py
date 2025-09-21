"""Core components of the demo trading bot.

The implementation focuses on modularity so that every part of the system can
be unit tested in isolation.  The modules defined here are used both by the
web application (``app.py``) and the automated tests.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional
import uuid

import httpx
import websockets

from .config import AppConfig, PathsConfig, StrategyConfig, WebSocketConfig, ensure_data_paths

logger = logging.getLogger(__name__)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _to_datetime(timestamp: Any) -> datetime:
    if isinstance(timestamp, datetime):
        return timestamp.astimezone(UTC)
    if isinstance(timestamp, (int, float)):
        # Accept both seconds and milliseconds.
        if timestamp > 1e12:
            timestamp = timestamp / 1000.0
        return datetime.fromtimestamp(float(timestamp), tz=UTC)
    if isinstance(timestamp, str):
        try:
            return _from_iso(timestamp)
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Cannot parse timestamp: {timestamp}") from exc
    raise TypeError(f"Unsupported timestamp type: {type(timestamp)!r}")


def simple_moving_average(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Candle:
    """Market candle representation."""

    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["open_time"] = _to_iso(self.open_time)
        data["close_time"] = _to_iso(self.close_time)
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Candle":
        return cls(
            open_time=_from_iso(payload["open_time"]),
            close_time=_from_iso(payload["close_time"]),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload.get("volume", 0.0)),
        )


@dataclass
class Trade:
    """Represents a trade tracked by the strategy."""

    id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    qty: float = 1.0
    take_profit: Optional[float] = None
    open_time: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    close_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: str = "active"
    trailing_stop_pct: Optional[float] = None
    profit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["open_time"] = _to_iso(self.open_time)
        if self.close_time:
            payload["close_time"] = _to_iso(self.close_time)
        if self.exit_price is not None:
            payload["exit_price"] = float(self.exit_price)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        open_time = _from_iso(data["open_time"]) if "open_time" in data else datetime.now(tz=UTC)
        close_time = _from_iso(data["close_time"]) if data.get("close_time") else None
        return cls(
            id=data["id"],
            symbol=data.get("symbol", "DEMO"),
            direction=data.get("direction", "long"),
            entry_price=float(data.get("entry_price", 0.0)),
            stop_loss=float(data.get("stop_loss", data.get("entry_price", 0.0))),
            qty=float(data.get("qty", 1.0)),
            take_profit=float(data["take_profit"]) if data.get("take_profit") is not None else None,
            open_time=open_time,
            close_time=close_time,
            exit_price=float(data["exit_price"]) if data.get("exit_price") is not None else None,
            status=data.get("status", "active"),
            trailing_stop_pct=float(data["trailing_stop_pct"]) if data.get("trailing_stop_pct") is not None else None,
            profit=float(data["profit"]) if data.get("profit") is not None else None,
        )

    def close(self, price: float, timestamp: datetime) -> None:
        self.status = "closed"
        self.exit_price = price
        self.close_time = timestamp
        self.profit = round((price - self.entry_price) * self.qty if self.direction == "long" else (self.entry_price - price) * self.qty, 8)


@dataclass
class Signal:
    id: str
    symbol: str
    direction: str
    price: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = _to_iso(self.timestamp)
        return data


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------


class DataManager:
    """Handles persistence of candles and trades on disk."""

    def __init__(self, candles_path: Path, trades_path: Path, max_candles: int = 500) -> None:
        self.candles_path = Path(candles_path)
        self.trades_path = Path(trades_path)
        ensure_data_paths(PathsConfig(data_dir=self.candles_path.parent, candles_file=self.candles_path, trades_file=self.trades_path))
        self._max_candles = max_candles
        self._lock = threading.RLock()
        self._symbol = "DEMO"
        self._interval = "1m"
        self._candles: List[Candle] = []
        self._active_trades: List[Trade] = []
        self._closed_trades: List[Trade] = []
        self._load()

    # -- loading -----------------------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            self._candles = self._load_candles()
            self._active_trades, self._closed_trades = self._load_trades()

    def _load_candles(self) -> List[Candle]:
        if not self.candles_path.exists():
            return []
        try:
            payload = json.loads(self.candles_path.read_text(encoding="utf-8"))
            self._symbol = payload.get("symbol", self._symbol)
            self._interval = payload.get("interval", self._interval)
            return [Candle.from_dict(item) for item in payload.get("candles", [])]
        except json.JSONDecodeError:
            logger.warning("Failed to decode candles.json; starting with empty list")
            return []

    def _load_trades(self) -> (List[Trade], List[Trade]):
        if not self.trades_path.exists():
            return [], []
        try:
            payload = json.loads(self.trades_path.read_text(encoding="utf-8"))
            active = [Trade.from_dict(item) for item in payload.get("active", [])]
            closed = [Trade.from_dict(item) for item in payload.get("closed", [])]
            return active, closed
        except json.JSONDecodeError:
            logger.warning("Failed to decode trades.json; starting empty")
            return [], []

    # -- candle operations --------------------------------------------------------
    def append_candle(self, candle: Candle) -> None:
        with self._lock:
            self._candles.append(candle)
            if len(self._candles) > self._max_candles:
                self._candles = self._candles[-self._max_candles :]
            self._persist_candles()

    def get_candles_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "symbol": self._symbol,
                "interval": self._interval,
                "candles": [candle.to_dict() for candle in self._candles],
            }

    def get_recent_candles(self, limit: int) -> List[Candle]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._candles[-limit:])

    def _persist_candles(self) -> None:
        payload = self.get_candles_payload()
        self.candles_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- trade operations ---------------------------------------------------------
    def add_trade(self, trade: Trade) -> Trade:
        with self._lock:
            self._active_trades.append(trade)
            self._persist_trades()
            return trade

    def active_trades(self) -> List[Trade]:
        with self._lock:
            return [Trade.from_dict(trade.to_dict()) for trade in self._active_trades]

    def _find_active_trade(self, trade_id: str) -> Optional[Trade]:
        for trade in self._active_trades:
            if trade.id == trade_id:
                return trade
        return None

    def update_trade(self, trade: Trade) -> None:
        with self._lock:
            for idx, existing in enumerate(self._active_trades):
                if existing.id == trade.id:
                    self._active_trades[idx] = trade
                    break
            self._persist_trades()

    def close_trade(self, trade_id: str, exit_price: float, timestamp: datetime) -> Optional[Trade]:
        with self._lock:
            trade = self._find_active_trade(trade_id)
            if not trade:
                return None
            trade.close(exit_price, timestamp)
            self._active_trades = [t for t in self._active_trades if t.id != trade_id]
            self._closed_trades.append(trade)
            self._persist_trades()
            return trade

    def get_trades_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": [trade.to_dict() for trade in self._active_trades],
                "closed": [trade.to_dict() for trade in self._closed_trades],
            }

    def _persist_trades(self) -> None:
        payload = self.get_trades_payload()
        self.trades_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Price processing
# ---------------------------------------------------------------------------


@dataclass
class _CandleBuilder:
    interval_seconds: int
    start_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float = 0.0

    def update(self, price: float, volume: float) -> None:
        self.close_price = price
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.volume += volume

    def to_candle(self) -> Candle:
        return Candle(
            open_time=self.start_time,
            close_time=self.start_time + timedelta(seconds=self.interval_seconds),
            open=self.open_price,
            high=self.high_price,
            low=self.low_price,
            close=self.close_price,
            volume=self.volume,
        )


class PriceProcessor:
    """Aggregates raw ticks into candles and handles Binance style klines."""

    def __init__(self, interval_seconds: int = 60) -> None:
        self.interval_seconds = interval_seconds
        self._builder: Optional[_CandleBuilder] = None

    def process_tick(self, message: Dict[str, Any]) -> Optional[Candle]:
        if "k" in message:
            return self._from_binance_kline(message["k"])
        return self._from_trade_tick(message)

    def _from_binance_kline(self, kline: Dict[str, Any]) -> Optional[Candle]:
        if not kline.get("x"):
            return None
        open_time = _to_datetime(kline["t"])
        close_time = _to_datetime(kline["T"])
        return Candle(
            open_time=open_time,
            close_time=close_time,
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline.get("v", 0.0)),
        )

    def _from_trade_tick(self, tick: Dict[str, Any]) -> Optional[Candle]:
        price = tick.get("price") or tick.get("p") or tick.get("close")
        if price is None:
            return None
        volume = tick.get("volume") or tick.get("q") or 0.0
        timestamp = tick.get("timestamp") or tick.get("E") or tick.get("event_time")
        if timestamp is None:
            return None
        price = float(price)
        volume = float(volume)
        ts = _to_datetime(timestamp)
        interval_start = ts - timedelta(seconds=ts.second % self.interval_seconds, microseconds=ts.microsecond)
        if not self._builder or interval_start != self._builder.start_time:
            candle = self.flush()
            self._builder = _CandleBuilder(
                interval_seconds=self.interval_seconds,
                start_time=interval_start,
                open_price=price,
                high_price=price,
                low_price=price,
                close_price=price,
                volume=volume,
            )
            return candle
        self._builder.update(price, volume)
        return None

    def flush(self) -> Optional[Candle]:
        if not self._builder:
            return None
        candle = self._builder.to_candle()
        self._builder = None
        return candle


# ---------------------------------------------------------------------------
# Strategy layer
# ---------------------------------------------------------------------------


class StrategyManager:
    """Generates trading signals and manages trade lifecycle."""

    def __init__(self, data_manager: DataManager, config: StrategyConfig) -> None:
        self.data_manager = data_manager
        self.config = config
        self._listeners: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def register_signal_listener(self, callback: Callable[[Signal], None]) -> None:
        self._listeners["signal"].append(callback)

    def register_trade_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners["trade"].append(callback)

    def _notify_signal(self, signal: Signal) -> None:
        for callback in self._listeners["signal"]:
            callback(signal)

    def _notify_trade(self, trade: Trade) -> None:
        for callback in self._listeners["trade"]:
            callback(trade.to_dict())

    def on_new_candle(self, candle: Candle) -> None:
        self._update_trailing_stops(candle)
        signal = self._maybe_generate_signal(candle)
        if signal:
            self._notify_signal(signal)
            trade = self._open_trade(signal, candle)
            if trade:
                self._notify_trade(trade)

    # -- signal generation -------------------------------------------------------
    def _maybe_generate_signal(self, candle: Candle) -> Optional[Signal]:
        needed = max(self.config.slow_ma, self.config.lookback)
        candles = self.data_manager.get_recent_candles(needed)
        if len(candles) < needed:
            return None
        closes = [c.close for c in candles]
        fast = simple_moving_average(closes[-self.config.fast_ma :]) if len(closes) >= self.config.fast_ma else None
        slow = simple_moving_average(closes[-self.config.slow_ma :]) if len(closes) >= self.config.slow_ma else None
        if fast is None or slow is None:
            return None
        direction: Optional[str] = None
        if fast > slow and not self._has_active_trade("long"):
            direction = "long"
        elif fast < slow and not self._has_active_trade("short"):
            direction = "short"
        if not direction:
            return None
        if candle.volume < self.config.min_volume:
            return None
        return Signal(id=str(uuid.uuid4()), symbol="DEMO", direction=direction, price=candle.close, timestamp=datetime.now(tz=UTC))

    def _has_active_trade(self, direction: str) -> bool:
        return any(trade.direction == direction for trade in self.data_manager.active_trades())

    def _open_trade(self, signal: Signal, candle: Candle) -> Optional[Trade]:
        stop_offset = candle.close * self.config.trailing_stop_pct
        if signal.direction == "long":
            stop_loss = candle.close - stop_offset
        else:
            stop_loss = candle.close + stop_offset
        trade = Trade(
            id=signal.id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=candle.close,
            stop_loss=stop_loss,
            trailing_stop_pct=self.config.trailing_stop_pct,
        )
        self.data_manager.add_trade(trade)
        return trade

    # -- trailing stop management -------------------------------------------------
    def _update_trailing_stops(self, candle: Candle) -> None:
        for trade in self.data_manager.active_trades():
            updated = False
            if trade.direction == "long":
                trail_value = candle.close * self.config.trailing_stop_pct
                new_stop = candle.close - trail_value
                if new_stop > trade.stop_loss:
                    trade.stop_loss = new_stop
                    updated = True
                if candle.low <= trade.stop_loss:
                    closed = self.data_manager.close_trade(trade.id, candle.close, candle.close_time)
                    if closed:
                        self._notify_trade(closed)
                    continue
            else:
                trail_value = candle.close * self.config.trailing_stop_pct
                new_stop = candle.close + trail_value
                if new_stop < trade.stop_loss:
                    trade.stop_loss = new_stop
                    updated = True
                if candle.high >= trade.stop_loss:
                    closed = self.data_manager.close_trade(trade.id, candle.close, candle.close_time)
                    if closed:
                        self._notify_trade(closed)
                    continue
            if updated:
                self.data_manager.update_trade(trade)


# ---------------------------------------------------------------------------
# WebSocket management
# ---------------------------------------------------------------------------


class WebSocketManager:
    """Connects to Binance (or a fake source) and dispatches messages."""

    def __init__(
        self,
        config: WebSocketConfig,
        message_handler: Callable[[Dict[str, Any]], Awaitable[None]],
        message_source: Optional[AsyncIterator[Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self._message_handler = message_handler
        self._message_source = message_source
        self._stop_event = asyncio.Event()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def run(self) -> None:
        self._stop_event.clear()
        if self._message_source is not None:
            async for message in self._message_source:
                if self._stop_event.is_set():
                    break
                await self._message_handler(message)
            return
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.config.stream_url) as ws:
                    self._ws = ws
                    await self._subscribe(ws)
                    async for raw_message in ws:
                        if self._stop_event.is_set():
                            break
                        try:
                            data = json.loads(raw_message)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non JSON websocket message: %s", raw_message)
                            continue
                        await self._message_handler(data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network path
                logger.warning("WebSocket connection error: %s", exc)
                await asyncio.sleep(self.config.reconnect_delay)
            finally:
                self._ws = None

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol) -> None:
        payload = {
            "method": "SUBSCRIBE",
            "params": [f"{self.config.symbol.lower()}@kline_1m"],
            "id": 1,
        }
        await ws.send(json.dumps(payload))

    async def stop(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            await self._ws.close()


# ---------------------------------------------------------------------------
# High level trading bot orchestration
# ---------------------------------------------------------------------------


class TradingBot:
    """High level orchestrator tying together the different components."""

    def __init__(
        self,
        config: AppConfig,
        data_manager: Optional[DataManager] = None,
        message_source: Optional[AsyncIterator[Dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        ensure_data_paths(config.paths)
        self.data_manager = data_manager or DataManager(config.paths.candles_file, config.paths.trades_file)
        self.price_processor = PriceProcessor(interval_seconds=config.websocket.price_interval_seconds)
        self.strategy_manager = StrategyManager(self.data_manager, config.strategy)
        self.websocket_manager = WebSocketManager(config.websocket, self._on_message, message_source=message_source)
        self._listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self.strategy_manager.register_signal_listener(lambda signal: self._notify("signal", signal.to_dict()))
        self.strategy_manager.register_trade_listener(lambda trade: self._notify("trade", trade))

    def register_listener(self, event: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners[event].append(callback)

    def _notify(self, event: str, payload: Dict[str, Any]) -> None:
        for callback in self._listeners[event]:
            callback(payload)

    async def _on_message(self, message: Dict[str, Any]) -> None:
        candle = self.price_processor.process_tick(message)
        if candle:
            self._handle_candle(candle)

    def _handle_candle(self, candle: Candle) -> None:
        self.data_manager.append_candle(candle)
        self._notify("candle", candle.to_dict())
        self.strategy_manager.on_new_candle(candle)

    async def run(self) -> None:
        await self.websocket_manager.run()
        candle = self.price_processor.flush()
        if candle:
            self._handle_candle(candle)

    async def stop(self) -> None:
        await self.websocket_manager.stop()
        candle = self.price_processor.flush()
        if candle:
            self._handle_candle(candle)

    # Convenience accessors -------------------------------------------------
    def get_candles(self) -> Dict[str, Any]:
        return self.data_manager.get_candles_payload()

    def get_trades(self) -> Dict[str, Any]:
        return self.data_manager.get_trades_payload()


# ---------------------------------------------------------------------------
# Optional bootstrap helper
# ---------------------------------------------------------------------------


class HistoricalBootstrapper:
    """Optionally prime the system with historical candles via HTTP."""

    def __init__(self, http_client: Optional[httpx.Client] = None) -> None:
        self.client = http_client or httpx.Client(timeout=5.0)

    def fetch_klines(self, base_url: str, symbol: str, interval: str = "1m", limit: int = 50) -> List[Dict[str, Any]]:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = self.client.get(f"{base_url}/api/v3/klines", params=params)
        response.raise_for_status()
        klines = response.json()
        parsed: List[Dict[str, Any]] = []
        for item in klines:
            parsed.append(
                {
                    "open_time": _to_datetime(item[0]),
                    "close_time": _to_datetime(item[6]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        return parsed

    def bootstrap(self, data_manager: DataManager, candles: List[Dict[str, Any]]) -> None:
        for raw in candles:
            candle = Candle(
                open_time=raw["open_time"],
                close_time=raw["close_time"],
                open=raw["open"],
                high=raw["high"],
                low=raw["low"],
                close=raw["close"],
                volume=raw.get("volume", 0.0),
            )
            data_manager.append_candle(candle)


# ---------------------------------------------------------------------------
# Background runner for the web application
# ---------------------------------------------------------------------------


class TradingBotService:
    """Runs the trading bot inside its own event loop thread."""

    def __init__(self, bot: TradingBot) -> None:
        self.bot = bot
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._task: Optional[asyncio.Future[Any]] = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(self.bot.run(), self.loop)

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        asyncio.run_coroutine_threadsafe(self.bot.stop(), self.loop).result(timeout=5)
        if self._task:
            self._task.cancel()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


__all__ = [
    "Candle",
    "Trade",
    "Signal",
    "DataManager",
    "PriceProcessor",
    "StrategyManager",
    "WebSocketManager",
    "TradingBot",
    "TradingBotService",
    "HistoricalBootstrapper",
]
