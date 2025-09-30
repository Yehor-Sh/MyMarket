"""Main orchestrator service combining market data, strategies and UI."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import queue
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

from binance_client import BinanceClient, Kline
from module_base import ModuleBase, Signal
from modules.cluster_engine import ClusterEngine
from multi_factor_engine import DEFAULT_FACTORS, FactorCallable, MultiFactorEngine


def _normalise_strategies(
    metadata: Optional[Dict[str, object]], fallback: Optional[str]
) -> List[str]:
    if metadata is None:
        strategies: Iterable[str] = []
    else:
        raw = metadata.get("strategies") if isinstance(metadata, dict) else None
        if isinstance(raw, list):
            strategies = [str(item).upper() for item in raw if item]
        else:
            strategies = []
    if not strategies and fallback:
        return [fallback.upper()]
    if not strategies:
        return []
    return sorted({strategy.upper() for strategy in strategies})
from module_worker import ModuleHealth, ModuleWorker


_logger = logging.getLogger(__name__)


MARKET_CONTEXT_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT"]


@dataclass
class Trade:
    trade_id: str
    symbol: str
    side: str
    strategy: str
    entry_price: float
    quantity: float
    opened_at: datetime
    trailing_percent: float
    trailing_stop: float
    initial_stop: float
    high_watermark: float
    low_watermark: float
    metadata: Dict[str, object] = field(default_factory=dict)
    confidence: float = 1.0
    current_price: Optional[float] = None
    max_profit_pct: float = 0.0
    current_profit_pct: float = 0.0
    exit_price: Optional[float] = None
    closed_at: Optional[datetime] = None
    profit_pct: Optional[float] = None
    is_active: bool = True

    def update_trailing_stop(self, price: float) -> Tuple[float, bool]:
        """Update trailing stop according to the latest ``price``."""

        if not self.is_active:
            return self.current_profit_pct, False

        self.current_price = price
        profit_pct = self._calculate_profit_pct(price)
        self.current_profit_pct = profit_pct
        closed = False

        if self.side == "LONG":
            if profit_pct > self.max_profit_pct:
                self.max_profit_pct = profit_pct
                if price > self.high_watermark:
                    self.high_watermark = price
                new_stop = price * (1 - self.trailing_percent / 100.0)
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
            if price <= self.trailing_stop:
                self._close(price)
                closed = True
        else:  # SHORT
            if profit_pct > self.max_profit_pct:
                self.max_profit_pct = profit_pct
                if price < self.low_watermark:
                    self.low_watermark = price
                new_stop = price * (1 + self.trailing_percent / 100.0)
                if new_stop < self.trailing_stop or self._is_initial_stop():
                    self.trailing_stop = new_stop
            if price >= self.trailing_stop:
                self._close(price)
                closed = True

        return self.current_profit_pct, closed

    def apply_price(self, price: float) -> bool:
        """Update trailing stop and close the trade if needed."""

        _, closed = self.update_trailing_stop(price)
        return closed

    def _close(self, price: float) -> None:
        self.exit_price = price
        self.closed_at = datetime.now().astimezone()
        self.profit_pct = self._calculate_profit_pct(price)
        self.current_profit_pct = self.profit_pct or 0.0
        self.max_profit_pct = max(self.max_profit_pct, self.current_profit_pct)
        self.current_price = price
        self.trailing_stop = price
        self.is_active = False

    def to_dict(self) -> Dict[str, object]:
        metadata = dict(self.metadata)
        strategies = _normalise_strategies(metadata, self.strategy)
        metadata.setdefault("strategies", strategies)
        metadata.setdefault("cluster_size", len(strategies))
        return {
            "id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "strategy": self.strategy,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "opened_at": self.opened_at.isoformat(),
            "trailing_stop": self.trailing_stop,
            "confidence": self.confidence,
            "metadata": metadata,
            "current_price": self.current_price,
            "current_profit_pct": self.current_profit_pct,
            "current_profit": self.current_profit_pct,
            "max_profit_pct": self.max_profit_pct,
            "max_profit": self.max_profit_pct,
            "exit_price": self.exit_price,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "profit_pct": self.profit_pct
            if self.profit_pct is not None
            else self.current_profit_pct,
            "status": "active" if self.is_active else "closed",
            "cluster_size": metadata.get("cluster_size"),
            "cluster_strategies": metadata.get("strategies"),
        }

    def _calculate_profit_pct(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry_price) / self.entry_price * 100.0
        return (self.entry_price - price) / self.entry_price * 100.0

    def _is_initial_stop(self) -> bool:
        tolerance = max(1e-8, self.entry_price * 1e-6)
        return abs(self.trailing_stop - self.initial_stop) <= tolerance


class Orchestrator:
    """Glue between market data, strategy modules and presentation layer."""

    def __init__(
        self,
        *,
        trailing_percent: float = 0.3,
        module_poll_interval: float = 60.0,
        max_closed_trades: int = 1000,
        max_tracked_symbols: int = 200,
        trend_mode: str = "global",       # <--- добавил
        trend_interval: str = "1m",       # <--- добавил
        trend_lookback: int = 20,
        cluster_threshold: int = 3,
        factor_min_pass: int = 4,
        factor_functions: Optional[Iterable[FactorCallable]] = None,
    ) -> None:
        self.client = BinanceClient()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.trailing_percent = trailing_percent
        self.module_poll_interval = module_poll_interval
        self.max_closed_trades = max_closed_trades
        self.max_tracked_symbols = max_tracked_symbols

        self.active_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self._symbol_index: Dict[str, set[str]] = defaultdict(set)
        self._duplicate_guard: Dict[Tuple[str, str, Tuple[str, ...]], str] = {}
        self._lock = threading.RLock()

        self.signal_queue: "queue.Queue[Signal]" = queue.Queue()
        self._signal_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.trend_mode = trend_mode
        self.trend_interval = trend_interval
        self.trend_lookback = trend_lookback

        self.strategies: Dict[str, ModuleBase] = self._discover_strategies()
        self.modules: List[ModuleBase] = [
            self.strategies[key] for key in sorted(self.strategies)
        ]
        self.workers: List[ModuleWorker] = []

        self.cluster_engine = ClusterEngine(cluster_threshold)
        factors = list(factor_functions) if factor_functions is not None else list(DEFAULT_FACTORS)
        self.multi_factor_engine = MultiFactorEngine(factors, min_pass=factor_min_pass)

        self.client.add_price_listener(self._handle_price_update)
        self._register_routes()
        self._register_socket_handlers()

        self.market_context: Dict[str, Dict[str, object]] = {}
        self._context_lock = threading.RLock()

    # ------------------------------------------------------------------
    def _register_routes(self) -> None:
        static_dir = Path(__file__).resolve().parent

        @self.app.route("/")
        def index():
            return send_from_directory(static_dir, "client.html")

    # ------------------------------------------------------------------
    def _register_socket_handlers(self) -> None:
        @self.socketio.on("connect")
        def _on_connect(auth=None):
            emit("trades", self._serialize_state())

        @self.socketio.on("request_state")
        def _on_request_state():
            emit("trades", self._serialize_state())

        @self.socketio.on("close_all_trades")
        def _on_close_all_trades(data=None):
            count = self.close_all_trades()
            return {"status": "ok", "count": count}

        @self.socketio.on("clear_cache")
        def _on_clear_cache(data=None):
            self.client.clear_caches()
            return {"status": "ok"}

        @self.socketio.on("request_module_health")
        def _on_request_module_health(data=None):
            health = self.get_modules_health()
            return {"status": "ok", "health": health}
        
        @self.socketio.on("set_trend_mode")
        def _on_set_trend_mode(data=None):
            mode = (data or {}).get("mode", "").lower()
            if mode not in ("global", "local"):
                return {"status": "error", "message": "invalid mode"}
            self.trend_mode = mode
            payload = self._broadcast_state()
            context = payload.get("context") if isinstance(payload, dict) else None
            return {"status": "ok", "mode": self.trend_mode, "context": context}


    # ------------------------------------------------------------------
    def _discover_strategies(self) -> Dict[str, ModuleBase]:
        """Load strategy implementations from the ``modules`` package."""

        try:
            import modules as strategies_pkg
        except ImportError:  # pragma: no cover - defensive
            _logger.exception("failed to import strategy package")
            return {}

        discovered: Dict[str, ModuleBase] = {}

        for module_info in pkgutil.iter_modules(strategies_pkg.__path__, prefix=f"{strategies_pkg.__name__}."):
            module_name = module_info.name
            short_name = module_name.rsplit(".", 1)[-1]
            if not short_name.startswith("strategy_"):
                continue
            try:
                module = importlib.import_module(module_name)
            except Exception:  # pragma: no cover - defensive
                _logger.exception("failed to import strategy module %s", module_name)
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module.__name__:
                    continue
                if not issubclass(obj, ModuleBase):
                    continue
                if inspect.isabstract(obj):
                    continue
                try:
                    instance = obj(self.client)
                except TypeError:
                    _logger.warning(
                        "strategy %s cannot be instantiated with the Binance client; skipping",
                        f"{module_name}.{obj.__name__}",
                    )
                    continue
                except Exception:  # pragma: no cover - defensive
                    _logger.exception(
                        "unexpected error initialising strategy %s",
                        f"{module_name}.{obj.__name__}",
                    )
                    continue

                if not hasattr(instance, "name") or not isinstance(instance.name, str):
                    _logger.warning(
                        "strategy %s does not define a valid name; skipping",
                        f"{module_name}.{obj.__name__}",
                    )
                    continue

                run_method = getattr(instance, "run", None)
                if not callable(run_method):
                    _logger.warning(
                        "strategy %s does not provide a callable 'run' method; skipping",
                        f"{module_name}.{obj.__name__}",
                    )
                    continue

                abbreviation = getattr(instance, "abbreviation", None)
                if not abbreviation:
                    _logger.warning(
                        "strategy %s does not declare an abbreviation; skipping",
                        f"{module_name}.{obj.__name__}",
                    )
                    continue

                key = str(abbreviation).upper()
                if key in discovered:
                    _logger.warning(
                        "duplicate strategy abbreviation '%s' found in %s; keeping existing instance",
                        key,
                        module_name,
                    )
                    continue

                discovered[key] = instance

        if not discovered:
            _logger.warning("no strategy modules were discovered")
        else:
            abbreviations = ", ".join(sorted(discovered))
            _logger.info(
                "discovered %d strategy module(s): %s",
                len(discovered),
                abbreviations,
            )

        return discovered

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.client.start()
        try:
            self.client.get_liquid_pairs(force_refresh=True)
        except Exception:
            # In environments without network access the orchestrator can still
            # operate using cached/synthetic data.
            pass
        self.client.subscribe_ticker(MARKET_CONTEXT_SYMBOLS)
        self._stop_event.clear()
        self._signal_thread = threading.Thread(target=self._signal_loop, daemon=True)
        self._signal_thread.start()
        self.workers = [
            ModuleWorker(
                module,
                self.signal_queue,
                symbols_provider=self._symbols_provider,
                interval=self.module_poll_interval,
            )
            for module in self.modules
        ]
        for worker in self.workers:
            worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.client.stop()
        if self._signal_thread and self._signal_thread.is_alive():
            self._signal_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    def _symbols_provider(self) -> Iterable[str]:
        pairs = self.client.get_liquid_pairs()
        base = pairs[: self.max_tracked_symbols] if self.max_tracked_symbols else pairs
        result = set(base) | {"BTCUSDT", "ETHUSDT"}
        self.client.subscribe_ticker(result)
        return list(result)


    # ------------------------------------------------------------------
    def _prepare_factor_inputs(
        self, signals: Iterable[Signal]
    ) -> Tuple[Dict[str, Sequence[Kline]], Dict[str, Dict[str, object]]]:
        symbols = {signal.symbol.upper() for signal in signals}
        candle_map: Dict[str, Sequence[Kline]] = {}
        for symbol in symbols:
            cached = self.client.get_cached_klines(symbol, "1h")
            if not cached:
                try:
                    cached = self.client.fetch_klines(symbol, "1h", 120)
                except Exception:  # pragma: no cover - defensive fallback
                    cached = []
            candle_map[symbol] = cached
        self._refresh_market_context()
        with self._context_lock:
            context = {
                ctx_symbol: dict(data)
                for ctx_symbol, data in self.market_context.items()
            }
        return candle_map, context


    # ------------------------------------------------------------------
    def _signal_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                signal = self.signal_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            raw_signals = [signal]
            while True:
                try:
                    raw_signals.append(self.signal_queue.get_nowait())
                except queue.Empty:
                    break
            clustered = self.cluster_engine.process_signals(raw_signals)
            if not clustered:
                continue
            if self.multi_factor_engine.has_factors:
                candles, context = self._prepare_factor_inputs(clustered)
            else:
                candles, context = {}, {}
            validated = self.multi_factor_engine.validate(clustered, candles, context)
            for filtered_signal in validated:
                self._handle_signal(filtered_signal)

    # ------------------------------------------------------------------
    def _handle_signal(self, signal: Signal) -> None:
        symbol = signal.symbol.upper()
        strategies = _normalise_strategies(signal.metadata, signal.strategy)
        key = (symbol, signal.side, tuple(strategies))
        allowed_symbols = set(self._symbols_provider())
        with self._lock:
            if key in self._duplicate_guard:
                return
        if symbol not in allowed_symbols:
            return
        price = self.client.get_price(symbol)
        if price is None:
            candles = self.client.fetch_klines(symbol, "1m", 1)
            if candles:
                price = candles[-1].close
        if price is None:
            primary_strategy = strategies[0] if strategies else signal.strategy
            module = self.strategies.get(primary_strategy.upper())
            if module is not None:
                cached_candles = self.client.get_cached_klines(symbol, module.interval)
                if cached_candles:
                    price = cached_candles[-1].close
        if price is None:
            return

        self.client.subscribe_ticker([symbol])

        initial_stop = self._initial_stop(price, signal.side)
        metadata = dict(signal.metadata)
        metadata.setdefault("strategies", strategies)
        metadata.setdefault("cluster_size", len(strategies))
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            side=signal.side,
            strategy=signal.strategy,
            entry_price=price,
            quantity=1.0,
            opened_at=datetime.now(),
            trailing_percent=self.trailing_percent,
            trailing_stop=initial_stop,
            initial_stop=initial_stop,
            high_watermark=price,
            low_watermark=price,
            metadata=metadata,
            confidence=signal.confidence,
            current_price=price,
        )
        with self._lock:
            self.active_trades[trade.trade_id] = trade
            self._symbol_index[symbol].add(trade.trade_id)
            self._duplicate_guard[key] = trade.trade_id
        self._broadcast_state()

    # ------------------------------------------------------------------
    def _initial_stop(self, entry_price: float, side: str) -> float:
        if side == "LONG":
            return entry_price * (1 - self.trailing_percent / 100.0)
        return entry_price * (1 + self.trailing_percent / 100.0)

    # ------------------------------------------------------------------
    def _handle_price_update(self, symbol: str, price: float) -> None:
        with self._lock:
            trade_ids = list(self._symbol_index.get(symbol, ()))
            trades = [self.active_trades[tid] for tid in trade_ids if tid in self.active_trades]
        updates: List[Dict[str, object]] = []
        closed: List[Trade] = []
        for trade in trades:
            _, was_closed = trade.update_trailing_stop(price)
            updates.append(trade.to_dict())
            if was_closed:
                closed.append(trade)

        context_payload: Optional[Dict[str, Dict[str, object]]] = None
        if symbol in MARKET_CONTEXT_SYMBOLS:
            self._refresh_market_context()
            with self._context_lock:
                context_payload = {
                    ctx_symbol: dict(data)
                    for ctx_symbol, data in self.market_context.items()
                }

        closed_payloads: List[Dict[str, object]] = []
        if closed:
            with self._lock:
                for trade in closed:
                    closed_payloads.append(self._finalize_trade(trade))
        for payload in closed_payloads:
            self.socketio.emit("trade_closed", payload)

        if updates or context_payload is not None:
            payload: Dict[str, object] = {"trades": updates}
            if context_payload is not None:
                payload["context"] = context_payload
            self.socketio.emit("prices_updated", payload)

        if closed_payloads:
            self._broadcast_state()
        elif trades:
            # No trade closed but trailing stop may have moved.
            self._broadcast_state()

    # ------------------------------------------------------------------
    def _finalize_trade(self, trade: Trade) -> Dict[str, object]:
        trade.is_active = False
        self.active_trades.pop(trade.trade_id, None)
        symbol_trades = self._symbol_index[trade.symbol]
        symbol_trades.discard(trade.trade_id)
        if not symbol_trades:
            self._symbol_index.pop(trade.symbol, None)
        strategies = _normalise_strategies(trade.metadata, trade.strategy)
        key = (trade.symbol, trade.side, tuple(strategies))
        self._duplicate_guard.pop(key, None)
        self.closed_trades.append(trade)
        if len(self.closed_trades) > self.max_closed_trades:
            self.closed_trades[:] = self.closed_trades[-self.max_closed_trades :]
        return trade.to_dict()

    # ------------------------------------------------------------------
    def _broadcast_state(self) -> Dict[str, object]:
        payload = self._serialize_state()
        self.socketio.emit("trades", payload)
        return payload

    # ------------------------------------------------------------------
    def close_all_trades(self) -> int:
        with self._lock:
            trades = list(self.active_trades.values())
        if not trades:
            return 0

        closed_payloads: List[Dict[str, object]] = []
        for trade in trades:
            price = trade.current_price
            if price is None:
                price = self.client.get_price(trade.symbol)
            if price is None:
                candles = self.client.fetch_klines(trade.symbol, "1m", 1)
                if candles:
                    price = candles[-1].close
            if price is None:
                price = trade.entry_price
            trade._close(price)
        with self._lock:
            for trade in trades:
                closed_payloads.append(self._finalize_trade(trade))

        if closed_payloads:
            for payload in closed_payloads:
                self.socketio.emit("trade_closed", payload)
            self._broadcast_state()
        return len(closed_payloads)

    # ------------------------------------------------------------------
    def _serialize_state(self) -> Dict[str, object]:
        self._refresh_market_context()
        with self._lock:
            active = [trade.to_dict() for trade in self.active_trades.values()]
            closed = [trade.to_dict() for trade in self.closed_trades]
        active.sort(key=lambda x: x["opened_at"], reverse=True)
        closed.sort(key=lambda x: x.get("closed_at") or "", reverse=True)
        with self._context_lock:
            context = {symbol: dict(data) for symbol, data in self.market_context.items()}
        return {
            "active": active,
            "closed": closed,
            "server_time": datetime.now().astimezone().isoformat(),
            "context": context,
        }


    # ------------------------------------------------------------------
    def get_modules_health(self) -> List[Dict[str, object]]:
        """Return health snapshots for all registered modules."""

        if not self.workers:
            return [
                ModuleHealth(
                    name=module.name,
                    abbreviation=module.abbreviation,
                    interval=module.interval,
                    lookback=int(getattr(module, "lookback", 0) or 0),
                    status="offline",
                ).to_dict(is_alive=False)
                for module in self.modules
            ]

        health: List[Dict[str, object]] = []
        for module, worker in zip_longest(self.modules, self.workers):
            if module is None:
                continue
            if worker is None:
                placeholder = ModuleHealth(
                    name=module.name,
                    abbreviation=module.abbreviation,
                    interval=module.interval,
                    lookback=int(getattr(module, "lookback", 0) or 0),
                    status="offline",
                )
                health.append(placeholder.to_dict(is_alive=False))
                continue
            snapshot = worker.get_health()
            health.append(snapshot.to_dict(is_alive=worker.is_alive()))
        return health
    
    def _refresh_market_context(self) -> None:
        context: Dict[str, Dict[str, object]] = {}
        if self.trend_mode == "global":
            snapshot = self.client.get_market_snapshot(MARKET_CONTEXT_SYMBOLS)
            for symbol in MARKET_CONTEXT_SYMBOLS:
                entry = snapshot.get(symbol, {})
                price = entry.get("price") if isinstance(entry, dict) else None
                change = entry.get("percent_change") if isinstance(entry, dict) else None
                trend = "FLAT"
                if change is not None:
                    if change > 0:
                        trend = "UP"
                    elif change < 0:
                        trend = "DOWN"
                context[symbol] = {"price": price, "trend": trend}

        else:  # локальный режим
            for symbol in MARKET_CONTEXT_SYMBOLS:
                try:
                    candles = self.client.fetch_klines(symbol, self.trend_interval, self.trend_lookback)
                    if not candles:
                        continue
                    last_close = candles[-1].close
                    first_close = candles[0].close
                    change_pct = (last_close - first_close) / first_close * 100

                    if abs(change_pct) < 0.05:  # меньше 0.05% → флэт
                        trend = "FLAT"
                    elif change_pct > 0:
                        trend = "UP"
                    else:
                        trend = "DOWN"

                    context[symbol] = {"price": last_close, "trend": trend}
                except Exception:
                    continue

        with self._context_lock:
            self.market_context = context



def create_app() -> Tuple[Flask, Orchestrator, SocketIO]:
    orchestrator = Orchestrator()
    orchestrator.start()
    return orchestrator.app, orchestrator, orchestrator.socketio


if __name__ == "__main__":  # pragma: no cover - manual execution
    app, orchestrator, socketio = create_app()
    try:
        socketio.run(app, host="0.0.0.0", port=8080)
    finally:
        orchestrator.stop()
