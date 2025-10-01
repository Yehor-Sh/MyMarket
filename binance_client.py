"""Binance market data client.

This module provides a thin abstraction over the Binance REST API that keeps
local caches of ticker prices and kline (candlestick) data.  The implementation
is purposely conservative – it relies on REST requests with lightweight
background threads instead of long running WebSocket connections so that it can
run in constrained environments (such as automated tests) without special
setup.  The public surface mirrors what strategy modules and the orchestrator
expect according to the architecture specification:

* a stream of ticker prices for subscribed symbols;
* a cache of recent klines per symbol/interval combination;
* helper utilities for filtering liquid trading pairs.

When available, a single background thread fetches ticker prices for all
subscribed symbols and notifies registered listeners.  Another thread keeps an
up-to-date list of liquid pairs by inspecting 24 hour statistics.  Both threads
are optional – the class can be used in a synchronous manner during unit tests.

The class is intentionally free of framework specific dependencies.  It only
relies on the standard library and ``requests`` which keeps the footprint very
small while remaining easy to mock.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote

import requests
import websocket

BINANCE_REST_ENDPOINT = "https://api.binance.com"
BINANCE_WS_ENDPOINT = "wss://stream.binance.com:9443/ws"

_logger = logging.getLogger(__name__)

MIN_VOLUME = 10_000_000.0
MIN_PRICE = 0.50
MAX_PRICE = 5000.0
EXCLUDE_STABLE = ("USDC", "BUSD", "FDUSD", "TUSD", "USDTUSDT")
EXCLUDE_SYMBOLS = {"BTCDOMUSDT", "SPXUSDT"}


@dataclass(frozen=True)
class Kline:
    """Represents a single Binance kline/candlestick entry."""

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

    @classmethod
    def from_rest(cls, payload: Sequence[float]) -> "Kline":
        """Create a :class:`Kline` instance from a REST API payload."""

        return cls(
            open_time=int(payload[0]),
            open=float(payload[1]),
            high=float(payload[2]),
            low=float(payload[3]),
            close=float(payload[4]),
            volume=float(payload[5]),
            close_time=int(payload[6]),
        )

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


class BinanceClient:
    """A light-weight Binance market data client.

    Parameters
    ----------
    ticker_interval : float, optional
        Interval (in seconds) between ticker refreshes.  Defaults to ``1.0``.
    liquidity_refresh_interval : float, optional
        Interval (in seconds) between liquidity snapshots.  Defaults to ``600``.
    min_volume : float, optional
        Minimal 24h quote volume required for a pair to be considered liquid.
    min_price : float, optional
        Minimal last price (in quote currency) to be considered liquid.
    max_price : float, optional
        Maximum last price to be considered liquid.
    session : requests.Session, optional
        Custom HTTP session; useful for injecting mocks during tests.
    offline_pairs : Sequence[str], optional
        Pre-defined list of symbols to use when a live refresh fails and the
        cache is empty.  Defaults to ``("BTCUSDT", "ETHUSDT")``.
    """

    DEFAULT_EXCLUDED_KEYWORDS = ("UP", "DOWN", "BULL", "BEAR")
    DEFAULT_EXCLUDED_BASES = {"USDT", "DAI", "EUR", "GBP"}.union(
        coin for coin in EXCLUDE_STABLE if len(coin) <= 5
    )
    DEFAULT_OFFLINE_PAIRS: Sequence[str] = ("BTCUSDT", "ETHUSDT")

    def __init__(
        self,
        *,
        ticker_interval: float = 1.0,
        liquidity_refresh_interval: float = 600.0,
        min_volume: float = MIN_VOLUME,
        min_price: float = MIN_PRICE,
        max_price: float = MAX_PRICE,
        session: Optional[requests.Session] = None,
        offline_pairs: Optional[Sequence[str]] = None,
    ) -> None:
        self._session = session or requests.Session()
        self._ticker_interval = ticker_interval
        self._liquidity_refresh_interval = liquidity_refresh_interval
        self._min_volume = float(min_volume)
        self._min_price = float(min_price)
        self._max_price = float(max_price)

        self._price_cache: Dict[str, float] = {}
        self._price_lock = threading.RLock()
        self._klines_cache: Dict[Tuple[str, str], List[Kline]] = {}
        self._klines_lock = threading.RLock()
        self._liquid_pairs: List[str] = []
        self._liquidity_lock = threading.RLock()
        self._last_liquidity_refresh: float = 0.0

        self._subscribed_symbols: Set[str] = set()
        self._kline_subscriptions: Set[Tuple[str, str]] = set()
        self._subscription_lock = threading.RLock()
        self._price_callbacks: List[Callable[[str, float], None]] = []
        self._callbacks_lock = threading.RLock()

        self._stop_event = threading.Event()
        self._ticker_thread: Optional[threading.Thread] = None
        self._liquidity_thread: Optional[threading.Thread] = None
        self._ws_stop_event = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_connected = threading.Event()
        self._ws_lock = threading.RLock()
        self._ws_app: Optional[websocket.WebSocketApp] = None
        self._ws_streams: Set[str] = set()
        self._ws_pending: Deque[str] = deque()
        self._ws_pending_set: Set[str] = set()
        self._ws_request_id = 0
        self._offline_pairs: List[str] = (
            list(offline_pairs)
            if offline_pairs is not None
            else list(self.DEFAULT_OFFLINE_PAIRS)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start background tasks.

        The method is idempotent: calling it multiple times will simply ensure
        that the threads are running.  Threads are flagged as daemon threads so
        that they will not prevent a Python interpreter from shutting down.
        """

        self._stop_event.clear()
        self._ws_stop_event.clear()
        if not (self._ticker_thread and self._ticker_thread.is_alive()):
            self._ticker_thread = threading.Thread(
                target=self._ticker_loop,
                name="BinanceTickerThread",
                daemon=True,
            )
            self._ticker_thread.start()

        if not (self._liquidity_thread and self._liquidity_thread.is_alive()):
            self._liquidity_thread = threading.Thread(
                target=self._liquidity_loop,
                name="BinanceLiquidityThread",
                daemon=True,
            )
            self._liquidity_thread.start()

        if not (self._ws_thread and self._ws_thread.is_alive()):
            self._ws_thread = threading.Thread(
                target=self._ws_loop,
                name="BinanceWebSocketThread",
                daemon=True,
            )
            self._ws_thread.start()

    def stop(self) -> None:
        """Stop all background threads and wait for them to terminate."""

        self._stop_event.set()
        self._ws_stop_event.set()
        if self._ticker_thread and self._ticker_thread.is_alive():
            self._ticker_thread.join(timeout=2.0)
        if self._liquidity_thread and self._liquidity_thread.is_alive():
            self._liquidity_thread.join(timeout=2.0)
        with self._ws_lock:
            ws_app = self._ws_app
        if ws_app is not None:
            try:
                ws_app.close()
            except Exception:  # pragma: no cover - defensive
                pass
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)
        self._ws_connected.clear()
        with self._ws_lock:
            self._ws_pending.clear()
            self._ws_pending_set.clear()

    def clear_caches(self) -> None:
        """Clear local price, kline and liquidity caches."""

        with self._price_lock:
            self._price_cache.clear()
        with self._klines_lock:
            self._klines_cache.clear()
        with self._liquidity_lock:
            self._liquid_pairs.clear()
            self._last_liquidity_refresh = 0.0

    # ------------------------------- Price handling -------------------
    def subscribe_ticker(self, symbols: Iterable[str]) -> None:
        """Subscribe the ticker updater to additional symbols."""

        normalized = {symbol.upper() for symbol in symbols if symbol}
        if not normalized:
            return
        with self._subscription_lock:
            new_symbols = normalized - self._subscribed_symbols
            if not new_symbols:
                return
            self._subscribed_symbols.update(new_symbols)
        streams = [f"{symbol.lower()}@miniTicker" for symbol in new_symbols]
        self._queue_ws_subscription(streams)

    def subscribe_klines(self, symbol: str, interval: str) -> None:
        """Subscribe to kline updates for ``symbol``/``interval``."""

        normalized_symbol = symbol.upper()
        normalized_interval = str(interval)
        if not normalized_symbol or not normalized_interval:
            return
        key = (normalized_symbol, normalized_interval)
        with self._subscription_lock:
            if key in self._kline_subscriptions:
                return
            self._kline_subscriptions.add(key)
        stream = f"{normalized_symbol.lower()}@kline_{normalized_interval}"
        self._queue_ws_subscription([stream])

    def add_price_listener(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback invoked whenever a ticker update is available."""

        with self._callbacks_lock:
            self._price_callbacks.append(callback)

    def get_price(self, symbol: str) -> Optional[float]:
        """Return the last known price for ``symbol`` (if any)."""

        with self._price_lock:
            return self._price_cache.get(symbol.upper())

    # ------------------------------- Kline handling -------------------
    def fetch_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Kline]:
        """Fetch ``limit`` klines for ``symbol`` and cache the result."""

        key = (symbol.upper(), interval)
        try:
            params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
            response = self._session.get(
                f"{BINANCE_REST_ENDPOINT}/api/v3/klines",
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            klines = [Kline.from_rest(entry) for entry in payload]
            with self._klines_lock:
                self._klines_cache[key] = klines
            self.subscribe_ticker([symbol.upper()])
            self.subscribe_klines(symbol, interval)
            return self.get_cached_klines(symbol, interval)
        except Exception as exc:  # pragma: no cover - network failure fallback
            _logger.warning("Failed to fetch klines for %s: %s", symbol, exc)
            with self._klines_lock:
                return list(self._klines_cache.get(key, []))

    def get_cached_klines(self, symbol: str, interval: str) -> List[Kline]:
        """Return a copy of cached klines for ``symbol``/``interval`` if present."""

        key = (symbol.upper(), interval)
        with self._klines_lock:
            return list(self._klines_cache.get(key, []))

    def get_market_snapshot(
        self, symbols: Sequence[str]
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Return last price and 24h percentage change for ``symbols``.

        The data is sourced from the ``/api/v3/ticker/24hr`` endpoint.  The
        helper gracefully degrades by falling back to the local price cache
        when a network error occurs so tests without network access keep
        functioning.
        """

        normalized = [symbol.upper() for symbol in symbols if symbol]
        if not normalized:
            return {}

        snapshot: Dict[str, Dict[str, Optional[float]]] = {}
        try:
            response = self._session.get(
                f"{BINANCE_REST_ENDPOINT}/api/v3/ticker/24hr",
                params={"symbols": json.dumps(normalized, separators=(",", ":"))},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            entries = payload if isinstance(payload, list) else [payload]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                symbol = str(entry.get("symbol", "")).upper()
                if symbol not in normalized:
                    continue
                price = entry.get("lastPrice")
                change = entry.get("priceChangePercent")
                try:
                    price_value = float(price) if price is not None else None
                except (TypeError, ValueError):
                    price_value = None
                try:
                    change_value = float(change) if change is not None else None
                except (TypeError, ValueError):
                    change_value = None
                snapshot[symbol] = {
                    "price": price_value,
                    "percent_change": change_value,
                }
        except Exception as exc:  # pragma: no cover - network failure fallback
            _logger.warning("Failed to fetch market snapshot: %s", exc)

        if len(snapshot) != len(normalized):
            with self._price_lock:
                for symbol in normalized:
                    data = snapshot.setdefault(symbol, {})
                    if "price" not in data:
                        cached_price = self._price_cache.get(symbol)
                        data["price"] = cached_price
                    data.setdefault("percent_change", None)

        return snapshot

    # ------------------------------- Liquidity ------------------------
    def get_liquid_pairs(self, force_refresh: bool = False) -> List[str]:
        """Return a filtered list of liquid pairs.

        Parameters
        ----------
        force_refresh : bool
            When ``True`` a fresh snapshot is downloaded immediately.
        """

        now = time.time()
        should_refresh = (
            force_refresh
            or now - self._last_liquidity_refresh > self._liquidity_refresh_interval
        )
        if should_refresh:
            try:
                self._refresh_liquid_pairs()
            except Exception as exc:  # pragma: no cover - defensive network guard
                _logger.warning(
                    "liquidity refresh failed, using cached snapshot: %s", exc
                )
        with self._liquidity_lock:
            cached_pairs = list(self._liquid_pairs)

        if cached_pairs:
            return cached_pairs

        if should_refresh and not cached_pairs:
            _logger.warning(
                "liquidity cache empty, falling back to %d offline pairs", len(self._offline_pairs)
            )
        return list(self._offline_pairs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ticker_loop(self) -> None:
        """Periodically refresh ticker prices."""

        while not self._stop_event.is_set():
            symbols = self._get_subscribed_symbols()
            if not symbols:
                if self._stop_event.wait(self._ticker_interval):
                    break
                continue
            if self._ws_connected.is_set():
                if self._stop_event.wait(self._ticker_interval):
                    break
                continue
            try:
                data = self._fetch_ticker_prices(symbols)
                for symbol, price in data.items():
                    self._publish_price_update(symbol, price)
            except Exception:  # pragma: no cover - defensive
                _logger.exception("ticker refresh failed")
            if self._stop_event.wait(self._ticker_interval):
                break

    def _liquidity_loop(self) -> None:
        """Periodically refresh the list of liquid trading pairs."""

        while not self._stop_event.is_set():
            try:
                self._refresh_liquid_pairs()
            except Exception:  # pragma: no cover - defensive
                _logger.exception("liquidity refresh failed")
            if self._stop_event.wait(self._liquidity_refresh_interval):
                break

    def _get_subscribed_symbols(self) -> List[str]:
        with self._subscription_lock:
            return sorted(self._subscribed_symbols)

    def _get_price_callbacks(self) -> List[Callable[[str, float], None]]:
        with self._callbacks_lock:
            return list(self._price_callbacks)

    def _publish_price_update(self, symbol: str, price: float) -> None:
        symbol = symbol.upper()
        with self._price_lock:
            self._price_cache[symbol] = price
        callbacks = self._get_price_callbacks()
        for callback in callbacks:
            try:
                callback(symbol, price)
            except Exception:  # pragma: no cover - defensive
                _logger.exception("price listener failed for %s", symbol)

    def _fetch_ticker_prices(self, symbols: Iterable[str]) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        symbols_list = [s.upper() for s in symbols if s and s.endswith("USDT")]

        if not symbols_list:
            return payload

        batches: List[List[str]] = []
        current_batch: List[str] = []

        for symbol in symbols_list:
            candidate = current_batch + [symbol]
            encoded_length = len(quote(json.dumps(candidate, separators=(",", ":")), safe=""))
            if current_batch and encoded_length > 512:
                batches.append(current_batch)
                current_batch = [symbol]
            else:
                current_batch = candidate

        if current_batch:
            batches.append(current_batch)

        for chunk in batches:
            params = {"symbols": json.dumps(chunk, separators=(",", ":"))}
            try:
                response = self._session.get(
                    f"{BINANCE_REST_ENDPOINT}/api/v3/ticker/price",
                    params=params,
                    timeout=5,
                )
                response.raise_for_status()
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response else None
                if status_code == 400:
                    chunk_payload, skipped = self._fetch_chunk_with_fallback(chunk)
                    payload.update(chunk_payload)
                    if skipped:
                        _logger.warning(
                            "Skipping invalid Binance symbols: %s",
                            ", ".join(sorted(skipped)),
                        )
                    continue
                raise

            data = response.json()

            # Защита от неожиданных ответов
            if isinstance(data, list):
                for entry in data:
                    try:
                        payload[entry["symbol"].upper()] = float(entry["price"])
                    except (KeyError, ValueError, TypeError):
                        continue
        return payload

    def _fetch_chunk_with_fallback(
        self, chunk: Sequence[str]
    ) -> Tuple[Dict[str, float], List[str]]:
        """Fetch ticker prices symbol-by-symbol when a batch request fails."""

        chunk_payload: Dict[str, float] = {}
        skipped: List[str] = []
        for symbol in chunk:
            try:
                response = self._session.get(
                    f"{BINANCE_REST_ENDPOINT}/api/v3/ticker/price",
                    params={"symbol": symbol},
                    timeout=5,
                )
                response.raise_for_status()
            except requests.HTTPError:
                skipped.append(symbol)
                continue
            except requests.RequestException:
                skipped.append(symbol)
                continue

            try:
                entry = response.json()
                chunk_payload[symbol.upper()] = float(entry["price"])
            except (ValueError, KeyError, TypeError):
                skipped.append(symbol)

        return chunk_payload, skipped


    def _queue_ws_subscription(self, streams: Iterable[str]) -> None:
        if not streams:
            return
        with self._ws_lock:
            added = False
            for stream in streams:
                if stream not in self._ws_streams:
                    self._ws_streams.add(stream)
                    self._enqueue_pending_locked(stream)
                    added = True
        if added:
            self._drain_ws_pending()

    def _enqueue_pending_locked(self, stream: str) -> None:
        if stream not in self._ws_pending_set:
            self._ws_pending.append(stream)
            self._ws_pending_set.add(stream)

    def _next_ws_id_locked(self) -> int:
        self._ws_request_id += 1
        return self._ws_request_id

    def _drain_ws_pending(self) -> None:
        while True:
            with self._ws_lock:
                if not (
                    self._ws_connected.is_set()
                    and self._ws_app is not None
                    and self._ws_pending
                ):
                    return
                stream = self._ws_pending.popleft()
                self._ws_pending_set.discard(stream)
                ws_app = self._ws_app
                request_id = self._next_ws_id_locked()
            payload = json.dumps(
                {"method": "SUBSCRIBE", "params": [stream], "id": request_id}
            )
            try:
                ws_app.send(payload)
            except Exception:  # pragma: no cover - defensive
                with self._ws_lock:
                    self._enqueue_pending_locked(stream)
                self._ws_connected.clear()
                try:
                    ws_app.close()
                except Exception:  # pragma: no cover - defensive
                    pass
                return

    def _ws_loop(self) -> None:
        while not self._ws_stop_event.is_set():
            try:
                ws_app = websocket.WebSocketApp(
                    BINANCE_WS_ENDPOINT,
                    on_open=self._on_ws_open,
                    on_close=self._on_ws_close,
                    on_error=self._on_ws_error,
                    on_message=self._on_ws_message,
                )
                with self._ws_lock:
                    self._ws_app = ws_app
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:  # pragma: no cover - defensive
                _logger.exception("binance websocket loop failed")
            finally:
                with self._ws_lock:
                    self._ws_app = None
                self._ws_connected.clear()
            if self._ws_stop_event.wait(5.0):
                break

    def _on_ws_open(self, ws: websocket.WebSocketApp) -> None:
        self._ws_connected.set()
        with self._ws_lock:
            for stream in self._ws_streams:
                self._enqueue_pending_locked(stream)
        self._drain_ws_pending()

    def _on_ws_close(
        self,
        ws: websocket.WebSocketApp,
        close_status_code: Optional[int],
        close_msg: Optional[str],
    ) -> None:
        self._ws_connected.clear()

    def _on_ws_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        self._ws_connected.clear()
        _logger.warning("binance websocket error: %s", error)
        try:
            ws.keep_running = False
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            ws.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def _on_ws_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return
        if not isinstance(payload, dict):
            return
        if "result" in payload:
            return
        if "stream" in payload:
            data = payload.get("data")
            if isinstance(data, dict):
                self._process_ws_payload(data)
            return
        self._process_ws_payload(payload)

    def _process_ws_payload(self, payload: Dict[str, object]) -> None:
        kline_data = payload.get("k")
        if isinstance(kline_data, dict):
            symbol = (
                kline_data.get("s")
                or payload.get("s")
                or payload.get("symbol")
            )
            interval = kline_data.get("i")
            if not symbol or not interval:
                return
            try:
                open_time = int(kline_data.get("t"))
                close_time = int(kline_data.get("T"))
                open_price = float(kline_data.get("o"))
                high_price = float(kline_data.get("h"))
                low_price = float(kline_data.get("l"))
                close_price = float(kline_data.get("c"))
                volume = float(kline_data.get("v"))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return
            is_final = bool(kline_data.get("x"))
            symbol_key = str(symbol).upper()
            interval_key = str(interval)
            incoming = Kline(
                open_time=open_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                close_time=close_time,
            )
            key = (symbol_key, interval_key)
            with self._klines_lock:
                cached = list(self._klines_cache.get(key, ()))
                inserted = False
                for index, existing in enumerate(cached):
                    if existing.open_time == open_time:
                        if not is_final:
                            incoming = Kline(
                                open_time=open_time,
                                open=existing.open,
                                high=max(existing.high, incoming.high),
                                low=min(existing.low, incoming.low),
                                close=incoming.close,
                                volume=max(existing.volume, incoming.volume),
                                close_time=incoming.close_time,
                            )
                        cached[index] = incoming
                        inserted = True
                        break
                    if existing.open_time > open_time:
                        cached.insert(index, incoming)
                        inserted = True
                        break
                if not inserted:
                    cached.append(incoming)
                self._klines_cache[key] = cached
            self._publish_price_update(symbol_key, incoming.close)
            return

        symbol = payload.get("s") or payload.get("symbol")
        price_value = payload.get("c") or payload.get("p") or payload.get("price")
        if not symbol or price_value is None:
            return
        try:
            price = float(price_value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return
        self._publish_price_update(str(symbol), price)

    def _refresh_liquid_pairs(self) -> None:
        response = self._session.get(
            f"{BINANCE_REST_ENDPOINT}/api/v3/ticker/24hr",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        pairs: List[str] = []
        for entry in data:
            try:
                symbol = entry["symbol"].upper()
                if symbol in EXCLUDE_SYMBOLS or symbol in EXCLUDE_STABLE:
                    continue
                if not symbol.endswith("USDT"):
                    continue
                base = symbol[:-4]
                if base in EXCLUDE_STABLE or base in self.DEFAULT_EXCLUDED_BASES:
                    continue
                if any(keyword in symbol for keyword in self.DEFAULT_EXCLUDED_KEYWORDS):
                    continue
                last_price = float(entry.get("lastPrice", 0.0))
                quote_volume = float(entry.get("quoteVolume", 0.0))
                if quote_volume < self._min_volume:
                    continue
                if not (self._min_price <= last_price <= self._max_price):
                    continue
                pairs.append(symbol)
            except Exception:  # pragma: no cover - defensive parsing
                continue
        pairs.sort()
        with self._liquidity_lock:
            self._liquid_pairs = pairs
            self._last_liquidity_refresh = time.time()

    # ------------------------------------------------------------------
    # Utility helpers useful during tests or command line usage
    # ------------------------------------------------------------------
    def prime_price(self, symbol: str, price: float) -> None:
        """Inject a price into the local cache (primarily for tests)."""

        with self._price_lock:
            self._price_cache[symbol.upper()] = float(price)

    def prime_klines(self, symbol: str, interval: str, klines: Iterable[Kline]) -> None:
        """Inject klines into the local cache (primarily for tests)."""

        key = (symbol.upper(), interval)
        with self._klines_lock:
            self._klines_cache[key] = list(klines)


__all__ = ["BinanceClient", "Kline"]
