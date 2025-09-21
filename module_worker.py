"""Worker thread responsible for running strategy modules."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Iterable

from module_base import ModuleBase, Signal

_logger = logging.getLogger(__name__)


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

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - long running thread
        while not self._stop_event.is_set():
            start = time.monotonic()
            try:
                symbols = list(self.symbols_provider())
            except Exception:  # pragma: no cover - defensive
                _logger.exception(
                    "module %s failed to retrieve symbols", self.module.abbreviation
                )
                symbols = []
            if symbols:
                try:
                    signals = self.module.get_signals(symbols)
                    for signal in signals:
                        self.signal_queue.put(signal)
                except Exception:  # pragma: no cover - defensive
                    # Errors are intentionally swallowed so that a misbehaving
                    # strategy cannot terminate the orchestrator.  Logging the
                    # failure helps with debugging while keeping the system
                    # resilient.
                    _logger.exception("module %s failed to produce signals", self.module.abbreviation)
                    continue
            elapsed = time.monotonic() - start
            wait_for = max(0.0, self.interval - elapsed)
            if self._stop_event.wait(wait_for):
                break


__all__ = ["ModuleWorker"]
