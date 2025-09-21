"""Worker thread responsible for running strategy modules."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Dict, Iterable, Literal

from module_base import ModuleBase, Signal

_logger = logging.getLogger(__name__)


@dataclass
class ModuleHealth:
    """Lightweight snapshot describing the state of a module worker."""

    name: str
    abbreviation: str
    interval: str
    lookback: int
    status: Literal["idle", "running", "ok", "error", "offline"] = "idle"
    last_run: datetime | None = None
    last_success: datetime | None = None
    last_duration: float | None = None
    last_error: str | None = None
    error_count: int = 0
    last_signal_count: int = 0
    total_signals: int = 0

    def to_dict(self, *, is_alive: bool | None = None) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "abbreviation": self.abbreviation,
            "interval": self.interval,
            "lookback": self.lookback,
            "status": self.status,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_duration": self.last_duration,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "last_signal_count": self.last_signal_count,
            "total_signals": self.total_signals,
        }
        if is_alive is not None:
            payload["is_alive"] = bool(is_alive)
        return payload


class ModuleWorker(threading.Thread):
    """Run a strategy module inside an isolated thread."""

    def __init__(
        self,
        module: ModuleBase,
        signal_queue: "queue.Queue[Signal]",
        *,
        symbols_provider: Callable[[], Iterable[str]],
        interval: float,
    ) -> None:
        super().__init__(name=f"ModuleWorker-{module.abbreviation}", daemon=True)
        self.module = module
        self.signal_queue = signal_queue
        self.interval = max(1.0, float(interval))
        self.symbols_provider = symbols_provider
        self._stop_event = threading.Event()
        self._health_lock = threading.RLock()
        self._health = ModuleHealth(
            name=module.name,
            abbreviation=module.abbreviation,
            interval=module.interval,
            lookback=int(getattr(module, "lookback", 0) or 0),
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    def get_health(self) -> ModuleHealth:
        """Return a snapshot of the worker state."""

        with self._health_lock:
            return replace(self._health)

    # ------------------------------------------------------------------
    def _record_start(self, run_at: datetime) -> None:
        with self._health_lock:
            self._health.status = "running"
            self._health.last_run = run_at

    def _record_success(self, run_at: datetime, duration: float, signals: int) -> None:
        with self._health_lock:
            self._health.status = "ok"
            self._health.last_run = run_at
            self._health.last_success = run_at
            self._health.last_duration = duration
            self._health.last_error = None
            self._health.last_signal_count = signals
            self._health.total_signals += signals

    def _record_idle(self, run_at: datetime, duration: float) -> None:
        with self._health_lock:
            self._health.status = "idle"
            self._health.last_run = run_at
            self._health.last_success = run_at
            self._health.last_duration = duration
            self._health.last_error = None
            self._health.last_signal_count = 0

    def _record_error(self, run_at: datetime, duration: float, message: str) -> None:
        with self._health_lock:
            self._health.status = "error"
            self._health.last_run = run_at
            self._health.last_duration = duration
            self._health.last_error = message
            self._health.error_count += 1
            self._health.last_signal_count = 0

    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - long running thread
        while not self._stop_event.is_set():
            start = time.monotonic()
            run_at = datetime.utcnow()
            self._record_start(run_at)

            encountered_error = False
            produced_signals = 0

            try:
                symbols = list(self.symbols_provider())
            except Exception as exc:  # pragma: no cover - defensive
                _logger.exception(
                    "module %s failed to retrieve symbols", self.module.abbreviation
                )
                duration = time.monotonic() - start
                self._record_error(run_at, duration, f"symbols: {exc}")
                encountered_error = True
                symbols = []

            if symbols:
                try:
                    signals = list(self.module.get_signals(symbols))
                except Exception as exc:  # pragma: no cover - defensive
                    # Errors are intentionally swallowed so that a misbehaving
                    # strategy cannot terminate the orchestrator.  Logging the
                    # failure helps with debugging while keeping the system
                    # resilient.
                    _logger.exception(
                        "module %s failed to produce signals", self.module.abbreviation
                    )
                    duration = time.monotonic() - start
                    self._record_error(run_at, duration, str(exc))
                    encountered_error = True
                else:
                    produced_signals = len(signals)
                    for signal in signals:
                        self.signal_queue.put(signal)
                    duration = time.monotonic() - start
                    self._record_success(run_at, duration, produced_signals)

            if not symbols and not encountered_error:
                duration = time.monotonic() - start
                self._record_idle(run_at, duration)

            elapsed = time.monotonic() - start
            wait_for = max(0.0, self.interval - elapsed)
            if self._stop_event.wait(wait_for):
                break


__all__ = ["ModuleWorker", "ModuleHealth"]
