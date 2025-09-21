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
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

BINANCE_REST_ENDPOINT = "https://api.binance.com"

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
    """

    DEFAULT_EXCLUDED_KEYWORDS = ("UP", "DOWN", "BULL", "BEAR")
    DEFAULT_EXCLUDED_BASES = {"USDT", "DAI", "EUR", "GBP"}.union(
        coin for coin in EXCLUDE_STABLE if len(coin) <= 5
    )

    def __init__(
        self,
        *,
        ticker_interval: float = 1.0,
        liquidity_refresh_interval: float = 600.0,
        min_volume: float = MIN_VOLUME,
        min_price: float = MIN_PRICE,
        max_price: float = MAX_PRICE,
        session: Optional[requests.Session] = None,
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
        self._subscription_lock = threading.RLock()
        self._price_callbacks: List[Callable[[str, float], None]] = []
        self._callbacks_lock = threading.RLock()

        self._stop_event = threading.Event()
        self._ticker_thread: Optional[threading.Thread] = None
        self._liquidity_thread: Optional[threading.Thread] = None

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

    def stop(self) -> None:
        """Stop all background threads and wait for them to terminate."""

        self._stop_event.set()
        if self._ticker_thread and self._ticker_thread.is_alive():
            self._ticker_thread.join(timeout=2.0)
        if self._liquidity_thread and self._liquidity_thread.is_alive():
            self._liquidity_thread.join(timeout=2.0)

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

        with self._subscription_lock:
            self._subscribed_symbols.update(symbols)

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
            return klines
        except Exception as exc:  # pragma: no cover - network failure fallback
            _logger.warning("Failed to fetch klines for %s: %s", symbol, exc)
            with self._klines_lock:
                return list(self._klines_cache.get(key, []))

    # ------------------------------- Liquidity ------------------------
    def get_liquid_pairs(self, force_refresh: bool = False) -> List[str]:
        """Return a filtered list of liquid pairs.

        Parameters
        ----------
        force_refresh : bool
            When ``True`` a fresh snapshot is downloaded immediately.
        """

        now = time.time()
        if force_refresh or now - self._last_liquidity_refresh > self._liquidity_refresh_interval:
            self._refresh_liquid_pairs()
        with self._liquidity_lock:
            return list(self._liquid_pairs)

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
            try:
                data = self._fetch_ticker_prices(symbols)
                callbacks = self._get_price_callbacks()
                for symbol, price in data.items():
                    with self._price_lock:
                        self._price_cache[symbol] = price
                    for callback in callbacks:
                        try:
                            callback(symbol, price)
                        except Exception:  # pragma: no cover - defensive
                            _logger.exception("price listener failed for %s", symbol)
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

    def _fetch_ticker_prices(self, symbols: Iterable[str]) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        symbols_list = [s.upper() for s in symbols]
        # Binance supports requesting a batch of tickers by encoding the list as
        # JSON.  Falling back to sequential requests would unnecessarily
        # increase the number of round trips.
        params = {"symbols": json.dumps(symbols_list)}
        response = self._session.get(
            f"{BINANCE_REST_ENDPOINT}/api/v3/ticker/price",
            params=params,
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        for entry in data:
            symbol = entry["symbol"].upper()
            price = float(entry["price"])
            payload[symbol] = price
        return payload

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
