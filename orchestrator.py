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
from datetime import datetime, UTC
from itertools import zip_longest
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    get_args,
    get_origin,
)

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit

from backtest.engine import Backtester, BacktestResult, load_klines_from_csv
from binance_client import BinanceClient, Kline
from config import CONFIG
from module_base import ModuleBase, Signal
from module_worker import ModuleHealth, ModuleWorker


_logger = logging.getLogger(__name__)


class ModeAlreadySelectedError(RuntimeError):
    """Raised when the orchestrator mode has already been chosen."""


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
    metadata: Dict[str, float] = field(default_factory=dict)
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
        self.closed_at = datetime.utcnow()
        self.profit_pct = self._calculate_profit_pct(price)
        self.current_profit_pct = self.profit_pct or 0.0
        self.max_profit_pct = max(self.max_profit_pct, self.current_profit_pct)
        self.current_price = price
        self.trailing_stop = price
        self.is_active = False

    def to_dict(self) -> Dict[str, object]:
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
            "metadata": self.metadata,
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
        }

    def _calculate_profit_pct(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry_price) / self.entry_price * 100.0
        return (self.entry_price - price) / self.entry_price * 100.0

    def _is_initial_stop(self) -> bool:
        tolerance = max(1e-8, self.entry_price * 1e-6)
        return abs(self.trailing_stop - self.initial_stop) <= tolerance


@dataclass
class StrategyParameterDefinition:
    """Describes a configurable constructor argument for a strategy."""

    key: str
    label: str
    annotation: object
    default: object
    kind: inspect._ParameterKind
    value_type: Optional[Type[object]]
    input_type: str


@dataclass
class StrategyDefinition:
    """Holds metadata required to instantiate a strategy module."""

    abbreviation: str
    name: str
    description: str
    cls: Type[ModuleBase]
    parameters: Dict[str, StrategyParameterDefinition]
    parameter_order: List[str]


class Orchestrator:
    """Glue between market data, strategy modules and presentation layer."""

    def __init__(
        self,
        *,
        trailing_percent: float = 0.3,
        module_poll_interval: float = 60.0,
        max_closed_trades: int = 500,
        max_tracked_symbols: int = 50,
    ) -> None:
        self.client = BinanceClient()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.config = CONFIG
        self.default_mode = self._normalize_mode(self.config.get("mode")) or "live"
        self.mode: Optional[str] = None
        self.backtest_config: Dict[str, object] = dict(self.config.get("backtest", {}))
        raw_strategy_overrides = self.backtest_config.get("strategy_parameters")
        strategy_overrides: Dict[str, Dict[str, object]] = {}
        if isinstance(raw_strategy_overrides, Mapping):
            for key, value in raw_strategy_overrides.items():
                if not isinstance(value, Mapping):
                    continue
                abbreviation = str(key).upper()
                fields: Dict[str, object] = {}
                for field_key, field_value in value.items():
                    fields[str(field_key)] = field_value
                if fields:
                    strategy_overrides[abbreviation] = fields
        self.strategy_overrides: Dict[str, Dict[str, object]] = strategy_overrides
        self.backtest_config["strategy_parameters"] = self.strategy_overrides
        if isinstance(self.config.get("backtest"), MutableMapping):
            self.config["backtest"]["strategy_parameters"] = self.strategy_overrides
        self.backtest_result: Optional[BacktestResult] = None
        self.trailing_percent = trailing_percent
        self.module_poll_interval = module_poll_interval
        self.max_closed_trades = max_closed_trades
        self.max_tracked_symbols = max_tracked_symbols

        self.active_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self._symbol_index: Dict[str, set[str]] = defaultdict(set)
        self._duplicate_guard: Dict[Tuple[str, str, str], str] = {}
        self._lock = threading.RLock()
        self._mode_lock = threading.Lock()

        self.signal_queue: "queue.Queue[Signal]" = queue.Queue()
        self._signal_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.strategy_definitions: Dict[str, StrategyDefinition] = {}
        self.strategies: Dict[str, ModuleBase] = self._discover_strategies()
        self._sync_strategy_overrides()
        self.modules: List[ModuleBase] = [
            self.strategies[key] for key in sorted(self.strategies)
        ]
        self.workers: List[ModuleWorker] = []

        self.client.add_price_listener(self._handle_price_update)
        self._register_routes()
        self._register_socket_handlers()

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_mode(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        mode = str(value).strip().lower()
        if mode in {"live", "backtest"}:
            return mode
        return None

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

        @self.socketio.on("select_mode")
        def _on_select_mode(data=None):
            payload = data or {}
            mode = payload.get("mode")
            try:
                started = self._activate_mode(mode)
            except ValueError:
                return {"status": "error", "message": "Неизвестный режим."}
            except ModeAlreadySelectedError as exc:
                return {
                    "status": "error",
                    "message": str(exc),
                    "mode": self.mode,
                }
            except Exception:
                _logger.exception("failed to activate mode %s", mode)
                return {
                    "status": "error",
                    "message": "Не удалось запустить выбранный режим.",
                }
            return {"status": "ok", "mode": self.mode, "started": started}

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

        @self.socketio.on("request_strategy_config")
        def _on_request_strategy_config(data=None):
            modules = self._build_strategy_config_payload()
            return {"status": "ok", "modules": modules}

        @self.socketio.on("update_strategy_config")
        def _on_update_strategy_config(data=None):
            if not self.strategy_definitions:
                return {"status": "error", "message": "Стратегии недоступны."}

            payload = data or {}
            overrides_payload = payload.get("overrides")
            if overrides_payload is None:
                return {
                    "status": "error",
                    "message": "Не переданы изменения параметров.",
                }
            try:
                normalized = self._validate_strategy_overrides(overrides_payload)
            except ValueError as exc:
                return {"status": "error", "message": str(exc)}

            if not normalized:
                return {"status": "error", "message": "Изменения не обнаружены."}

            changed = self._apply_strategy_overrides(normalized)
            if not changed:
                return {"status": "error", "message": "Изменения не обнаружены."}

            modules = self._build_strategy_config_payload()

            backtest_result: Optional[BacktestResult] = None
            try:
                backtest_result = self.run_backtest_with_overrides()
            except Exception:
                _logger.exception(
                    "failed to run backtest after updating strategy parameters"
                )
                return {
                    "status": "error",
                    "message": "Параметры сохранены, но выполнить бэктест не удалось.",
                    "modules": modules,
                }

            response: Dict[str, Any] = {
                "status": "ok",
                "message": "Параметры обновлены.",
                "modules": modules,
            }
            if backtest_result:
                response["backtest"] = backtest_result.to_dict()
            return response

    # ------------------------------------------------------------------
    def _discover_strategies(self) -> Dict[str, ModuleBase]:
        """Load strategy implementations from the ``modules`` package."""

        try:
            import modules as strategies_pkg
        except ImportError:  # pragma: no cover - defensive
            _logger.exception("failed to import strategy package")
            return {}

        self.strategy_definitions.clear()
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
                try:
                    definition = self._build_strategy_definition(obj, instance)
                except Exception:  # pragma: no cover - defensive
                    _logger.exception(
                        "failed to extract metadata for strategy %s",
                        f"{module_name}.{obj.__name__}",
                    )
                else:
                    self.strategy_definitions[key] = definition

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
    def _build_strategy_definition(
        self,
        cls: Type[ModuleBase],
        instance: ModuleBase,
    ) -> StrategyDefinition:
        signature = inspect.signature(cls.__init__)
        parameters: Dict[str, StrategyParameterDefinition] = {}
        order: List[str] = []

        for name, parameter in signature.parameters.items():
            if name in {"self", "client"}:
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            annotation = parameter.annotation
            default = parameter.default
            value_type = self._resolve_parameter_annotation(annotation, default)
            label = self._humanize_parameter_name(name)
            input_type = self._infer_input_type(value_type)

            parameters[name] = StrategyParameterDefinition(
                key=name,
                label=label,
                annotation=annotation,
                default=default,
                kind=parameter.kind,
                value_type=value_type,
                input_type=input_type,
            )
            order.append(name)

        description = inspect.getdoc(cls) or ""
        if description:
            description = description.strip().splitlines()[0]

        return StrategyDefinition(
            abbreviation=instance.abbreviation,
            name=instance.name,
            description=description,
            cls=cls,
            parameters=parameters,
            parameter_order=order,
        )

    # ------------------------------------------------------------------
    def _resolve_parameter_annotation(
        self,
        annotation: object,
        default: object,
    ) -> Optional[Type[object]]:
        if annotation is inspect._empty or annotation is None:
            if default is inspect._empty:
                return None
            return type(default)

        if isinstance(annotation, str):
            text = annotation.strip()
            if not text:
                if default is inspect._empty:
                    return None
                return type(default)
            lowered = text.lower()
            optional_prefixes = ("optional[", "typing.optional[")
            for prefix in optional_prefixes:
                if lowered.startswith(prefix) and lowered.endswith("]"):
                    inner = text[text.find("[") + 1 : -1]
                    return self._resolve_parameter_annotation(inner, default)
            mapping = {
                "int": int,
                "float": float,
                "str": str,
                "string": str,
                "bool": bool,
                "boolean": bool,
            }
            resolved = mapping.get(lowered)
            if resolved:
                return resolved
            # Handle union-style annotations like "float | None"
            if "|" in text:
                parts = [part.strip() for part in text.split("|") if part.strip()]
                without_none = [part for part in parts if part.lower() != "none"]
                if len(without_none) == 1:
                    return self._resolve_parameter_annotation(without_none[0], default)
            return type(default) if default is not inspect._empty else None

        origin = get_origin(annotation)
        if origin:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(args) == 1:
                return self._resolve_parameter_annotation(args[0], default)
        if isinstance(annotation, type):
            return annotation

        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_input_type(value_type: Optional[Type[object]]) -> str:
        if value_type is bool:
            return "checkbox"
        if value_type in {int, float}:
            return "number"
        return "text"

    # ------------------------------------------------------------------
    @staticmethod
    def _humanize_parameter_name(name: str) -> str:
        if not name:
            return ""
        tokens = name.replace("-", "_").split("_")
        parts: List[str] = []
        for token in tokens:
            if not token:
                continue
            lowered = token.lower()
            if len(lowered) <= 4:
                parts.append(lowered.upper())
            else:
                parts.append(lowered.capitalize())
        return " ".join(parts) if parts else name

    # ------------------------------------------------------------------
    def _sync_strategy_overrides(self) -> None:
        if not self.strategy_definitions:
            return

        synchronized: Dict[str, Dict[str, object]] = {}
        for key, overrides in self.strategy_overrides.items():
            definition = self.strategy_definitions.get(key)
            if not definition or not isinstance(overrides, Mapping):
                continue

            module_overrides: Dict[str, object] = {}
            for field, raw_value in overrides.items():
                meta = definition.parameters.get(str(field))
                if not meta:
                    continue
                try:
                    coerced = self._coerce_parameter_value(meta, raw_value)
                except ValueError:
                    continue
                default_value = meta.default if meta.default is not inspect._empty else None
                if coerced == default_value:
                    continue
                module_overrides[meta.key] = coerced

            if module_overrides:
                synchronized[key] = module_overrides

        self.strategy_overrides = synchronized
        self.backtest_config["strategy_parameters"] = synchronized
        if isinstance(self.config.get("backtest"), MutableMapping):
            self.config["backtest"]["strategy_parameters"] = synchronized

    # ------------------------------------------------------------------
    def _build_strategy_config_payload(self) -> List[Dict[str, object]]:
        modules: List[Dict[str, object]] = []
        for key in sorted(self.strategy_definitions):
            definition = self.strategy_definitions[key]
            overrides = self.strategy_overrides.get(key, {})
            fields: List[Dict[str, object]] = []
            for param_name in definition.parameter_order:
                meta = definition.parameters.get(param_name)
                if not meta:
                    continue
                default_value = meta.default if meta.default is not inspect._empty else None
                value = overrides.get(param_name, default_value)
                field_payload: Dict[str, object] = {
                    "key": meta.key,
                    "label": meta.label,
                    "type": meta.input_type,
                    "value": value,
                }
                if meta.default is not inspect._empty:
                    field_payload["default"] = default_value
                fields.append(field_payload)

            module_payload: Dict[str, object] = {
                "key": definition.abbreviation,
                "identifier": definition.abbreviation,
                "abbreviation": definition.abbreviation,
                "name": definition.name,
                "fields": fields,
            }
            if definition.description:
                module_payload["description"] = definition.description
            modules.append(module_payload)

        return modules

    # ------------------------------------------------------------------
    def _coerce_parameter_value(
        self,
        meta: StrategyParameterDefinition,
        value: object,
    ) -> object:
        if value is None:
            return None

        expected = meta.value_type
        label = meta.label or meta.key

        if expected is None:
            return value

        if expected is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "y", "on"}:
                    return True
                if normalized in {"0", "false", "no", "n", "off"}:
                    return False
            raise ValueError(
                f"Поле «{label}» должно быть булевым значением (да/нет)."
            )

        if expected is int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(
                        f"Поле «{label}» должно быть целым числом."
                    )
                return int(value)
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    raise ValueError(
                        f"Поле «{label}» должно быть целым числом."
                    )
                try:
                    return int(text, 10)
                except ValueError:
                    try:
                        parsed = float(text)
                    except ValueError as exc:  # pragma: no cover - defensive
                        raise ValueError(
                            f"Поле «{label}» должно быть целым числом."
                        ) from exc
                    if not parsed.is_integer():
                        raise ValueError(
                            f"Поле «{label}» должно быть целым числом."
                        )
                    return int(parsed)
            raise ValueError(
                f"Поле «{label}» должно быть целым числом."
            )

        if expected is float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    raise ValueError(
                        f"Поле «{label}» должно быть числом."
                    )
                try:
                    return float(text)
                except ValueError as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Поле «{label}» должно быть числом."
                    ) from exc
            raise ValueError(f"Поле «{label}» должно быть числом.")

        if expected is str:
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    raise ValueError(
                        f"Поле «{label}» должно быть строкой."
                    )
                return text
            text = str(value)
            if not text:
                raise ValueError(f"Поле «{label}» должно быть строкой.")
            return text

        return value

    # ------------------------------------------------------------------
    def _validate_strategy_overrides(
        self,
        overrides: object,
    ) -> Dict[str, Dict[str, object]]:
        if not isinstance(overrides, Mapping):
            raise ValueError("Некорректные данные конфигурации стратегий.")

        sanitized: Dict[str, Dict[str, object]] = {}
        for raw_key, raw_fields in overrides.items():
            abbreviation = str(raw_key or "").upper().strip()
            if not abbreviation:
                continue
            definition = self.strategy_definitions.get(abbreviation)
            if not definition:
                raise ValueError(f"Стратегия «{raw_key}» не найдена.")
            if not isinstance(raw_fields, Mapping):
                raise ValueError(
                    f"Некорректные параметры для стратегии {definition.name}."
                )

            module_overrides: Dict[str, object] = {}
            for raw_field, raw_value in raw_fields.items():
                field_key = str(raw_field or "").strip()
                if not field_key:
                    continue
                meta = definition.parameters.get(field_key)
                if not meta:
                    raise ValueError(
                        f"Параметр «{field_key}» не поддерживается стратегией {definition.name}."
                    )

                if raw_value is None or (
                    isinstance(raw_value, str) and not raw_value.strip()
                ):
                    default_value = (
                        meta.default if meta.default is not inspect._empty else None
                    )
                    module_overrides[field_key] = default_value
                    continue

                coerced = self._coerce_parameter_value(meta, raw_value)
                module_overrides[field_key] = coerced

            if module_overrides:
                sanitized[abbreviation] = module_overrides

        return sanitized

    # ------------------------------------------------------------------
    def _apply_strategy_overrides(
        self,
        overrides: Dict[str, Dict[str, object]],
    ) -> bool:
        if not overrides:
            return False

        changed = False
        for key, module_values in overrides.items():
            definition = self.strategy_definitions.get(key)
            if not definition:
                continue

            existing = dict(self.strategy_overrides.get(key, {}))
            for field, value in module_values.items():
                meta = definition.parameters.get(field)
                if not meta:
                    continue
                default_value = meta.default if meta.default is not inspect._empty else None
                if value == default_value:
                    if field in existing:
                        existing.pop(field)
                        changed = True
                    continue
                if existing.get(field) != value:
                    existing[field] = value
                    changed = True

            if existing:
                self.strategy_overrides[key] = existing
            else:
                if key in self.strategy_overrides:
                    changed = True
                    self.strategy_overrides.pop(key, None)

        if changed:
            sanitized = {k: dict(v) for k, v in self.strategy_overrides.items() if v}
            self.strategy_overrides = sanitized
            self.backtest_config["strategy_parameters"] = sanitized
            if isinstance(self.config.get("backtest"), MutableMapping):
                self.config["backtest"]["strategy_parameters"] = sanitized

        return changed

    # ------------------------------------------------------------------
    def _instantiate_strategy(
        self,
        definition: StrategyDefinition,
        overrides: Optional[Mapping[str, object]] = None,
    ) -> ModuleBase:
        kwargs: Dict[str, object] = {}
        if overrides:
            for key, value in overrides.items():
                if value is inspect._empty:
                    continue
                kwargs[str(key)] = value
        return definition.cls(self.client, **kwargs)

    # ------------------------------------------------------------------
    def run_backtest_with_overrides(self) -> Optional[BacktestResult]:
        result = self._run_backtest()
        if result:
            self._broadcast_state()
        return result

    def start(self, mode: Optional[str] = None) -> None:
        target_mode = self._normalize_mode(mode) or self.default_mode
        if target_mode not in {"live", "backtest"}:
            _logger.warning(
                "Unsupported orchestrator mode '%s'; defaulting to 'live'", target_mode
            )
            target_mode = "live"
        try:
            started = self._activate_mode(target_mode)
        except ModeAlreadySelectedError as exc:
            _logger.warning("%s", exc)
            return
        except ValueError:
            _logger.warning("Unsupported orchestrator mode '%s'; unable to start", mode)
            return
        if not started:
            _logger.info("Orchestrator already running in %s mode", self.mode)

    # ------------------------------------------------------------------
    def _start_live_mode(self) -> None:
        self.backtest_result = None
        self.client.start()
        try:
            self.client.get_liquid_pairs(force_refresh=True)
        except Exception:
            # In environments without network access the orchestrator can still
            # operate using cached/synthetic data.
            pass
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

    # ------------------------------------------------------------------
    def _activate_mode(self, mode: Optional[str]) -> bool:
        normalized = self._normalize_mode(mode)
        if not normalized:
            raise ValueError(f"unsupported mode: {mode!r}")

        conflict: Optional[str] = None
        with self._mode_lock:
            if self.mode is None:
                self.mode = normalized
            elif self.mode == normalized:
                _logger.info("Mode %s already active", normalized)
                return False
            else:
                conflict = self.mode

        if conflict:
            raise ModeAlreadySelectedError(
                f"Режим уже выбран: {str(conflict).upper()}"
            )

        try:
            if normalized == "backtest":
                _logger.info("Starting orchestrator in backtest mode")
                result = self._run_backtest()
                self.backtest_result = result
            else:
                _logger.info("Starting orchestrator in live mode")
                self._start_live_mode()
        except Exception:
            with self._mode_lock:
                self.mode = None
            raise

        self._broadcast_state()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)
        self.client.stop()
        if self._signal_thread and self._signal_thread.is_alive():
            self._signal_thread.join(timeout=1.0)
        self.workers = []
        self._signal_thread = None
        self.backtest_result = None
        with self._mode_lock:
            self.mode = None

    # ------------------------------------------------------------------
    def _run_backtest(self) -> Optional[BacktestResult]:
        self.backtest_result = None
        if not self.strategy_definitions:
            _logger.warning("no strategy modules available for backtest")
            return None

        cfg = self.backtest_config
        strategy_key = str(cfg.get("strategy") or "").upper()
        definition = self.strategy_definitions.get(strategy_key)
        if not definition:
            available = sorted(self.strategy_definitions)
            if not available:
                _logger.warning("no strategy modules available for backtest")
                return None
            fallback_key = available[0]
            definition = self.strategy_definitions[fallback_key]
            if strategy_key:
                _logger.warning(
                    "strategy %s not found; defaulting to %s",
                    strategy_key,
                    definition.abbreviation,
                )
            strategy_key = definition.abbreviation
            cfg["strategy"] = strategy_key
            if isinstance(self.config.get("backtest"), MutableMapping):
                self.config["backtest"]["strategy"] = strategy_key

        overrides = self.strategy_overrides.get(strategy_key, {})
        try:
            strategy = self._instantiate_strategy(definition, overrides)
        except Exception:
            _logger.exception(
                "failed to instantiate strategy %s for backtest",
                definition.abbreviation,
            )
            return None

        symbol = str(cfg.get("symbol", "BTCUSDT")).upper()
        interval = str(cfg.get("interval") or getattr(strategy, "interval", "1h"))
        limit_value = cfg.get("limit")
        if limit_value is None:
            limit = int(max(strategy.lookback, strategy.minimum_bars))
        else:
            try:
                limit = int(limit_value)
            except (TypeError, ValueError):
                limit = int(max(strategy.lookback, strategy.minimum_bars))

        candles: List[Kline] = []
        csv_path = cfg.get("csv_path")
        if csv_path:
            try:
                candles = load_klines_from_csv(csv_path)
            except FileNotFoundError:
                _logger.error("backtest CSV %s not found", csv_path)
                return None
            except Exception:
                _logger.exception("failed to load backtest CSV %s", csv_path)
                return None

        if not candles:
            try:
                candles = self.client.fetch_klines(
                    symbol,
                    interval,
                    limit,
                    is_backtest=True,
                )
            except Exception:
                _logger.exception(
                    "failed to fetch historical klines for %s (%s)", symbol, interval
                )
                return None

        if not candles:
            _logger.error("no historical candles available for %s", symbol)
            return None

        trailing_percent = float(cfg.get("trailing_percent", self.trailing_percent))
        enable_trailing = bool(cfg.get("enable_trailing", True))
        initial_capital = float(cfg.get("initial_capital", 10_000.0))
        commission_pct = float(cfg.get("commission_pct", 0.0))
        report_directory = cfg.get("report_directory")

        try:
            backtester = Backtester(
                strategy,
                candles,
                symbol=symbol,
                trailing_percent=trailing_percent,
                enable_trailing=enable_trailing,
                initial_capital=initial_capital,
                commission_pct=commission_pct,
                report_directory=report_directory,
            )
            result = backtester.run()
        except Exception:
            _logger.exception("backtest execution failed for %s", symbol)
            return None

        self.backtest_result = result
        metrics = result.metrics
        _logger.info(
            "Backtest for %s (%s) completed: return %.2f%%, win rate %.2f%%, sharpe %.2f",
            symbol,
            strategy.abbreviation,
            metrics.get("total_return_pct", 0.0),
            metrics.get("win_rate_pct", 0.0),
            metrics.get("sharpe_ratio", 0.0),
        )
        _logger.info("Report stored in %s", result.report_directory)
        return result

    # ------------------------------------------------------------------
    def _symbols_provider(self) -> Iterable[str]:
        pairs = self.client.get_liquid_pairs()
        return pairs[: self.max_tracked_symbols]

    # ------------------------------------------------------------------
    def _signal_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                signal = self.signal_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._handle_signal(signal)

    # ------------------------------------------------------------------
    def _handle_signal(self, signal: Signal) -> None:
        symbol = signal.symbol.upper()
        key = (symbol, signal.side, signal.strategy)
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
            return

        self.client.subscribe_ticker([symbol])

        initial_stop = self._initial_stop(price, signal.side)
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            side=signal.side,
            strategy=signal.strategy,
            entry_price=price,
            quantity=1.0,
            opened_at=datetime.utcnow(),
            trailing_percent=self.trailing_percent,
            trailing_stop=initial_stop,
            initial_stop=initial_stop,
            high_watermark=price,
            low_watermark=price,
            metadata=signal.metadata,
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

        closed_payloads: List[Dict[str, object]] = []
        if closed:
            with self._lock:
                for trade in closed:
                    closed_payloads.append(self._finalize_trade(trade))
        for payload in closed_payloads:
            self.socketio.emit("trade_closed", payload)

        if updates:
            self.socketio.emit("prices_updated", {"trades": updates})

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
        key = (trade.symbol, trade.side, trade.strategy)
        self._duplicate_guard.pop(key, None)
        self.closed_trades.append(trade)
        if len(self.closed_trades) > self.max_closed_trades:
            self.closed_trades[:] = self.closed_trades[-self.max_closed_trades :]
        return trade.to_dict()

    # ------------------------------------------------------------------
    def _broadcast_state(self) -> None:
        payload = self._serialize_state()
        self.socketio.emit("trades", payload)

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
        server_time = datetime.now(UTC).isoformat()

        if self.mode == "backtest" and self.backtest_result:
            strategy_key = self.backtest_result.strategy
            closed_trades: List[Dict[str, object]] = []
            for trade in self.backtest_result.trades:
                payload = dict(trade)
                if not payload.get("strategy"):
                    payload["strategy"] = strategy_key
                if not payload.get("module"):
                    payload["module"] = strategy_key
                closed_trades.append(payload)
            return {
                "active": [],
                "closed": closed_trades,
                "server_time": server_time,
                "mode": self.mode or "backtest",
            }

        with self._lock:
            active = [trade.to_dict() for trade in self.active_trades.values()]
            closed = [trade.to_dict() for trade in self.closed_trades]
        active.sort(key=lambda x: x["opened_at"], reverse=True)
        closed.sort(key=lambda x: x.get("closed_at") or "", reverse=True)
        return {
            "active": active,
            "closed": closed,
            "server_time": server_time,
            "mode": self.mode,
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


def create_app() -> Tuple[Flask, Orchestrator, SocketIO]:
    orchestrator = Orchestrator()
    return orchestrator.app, orchestrator, orchestrator.socketio


if __name__ == "__main__":  # pragma: no cover - manual execution
    app, orchestrator, socketio = create_app()
    try:
        socketio.run(app, host="0.0.0.0", port=8080)
    finally:
        orchestrator.stop()
