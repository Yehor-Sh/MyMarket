"""Signal clustering engine for aggregating strategy confirmations."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from module_base import Signal


class ClusterEngine:
    """Aggregate raw strategy signals into cluster confirmations."""

    def __init__(self, min_cluster_size: int = 3) -> None:
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1")
        self.min_cluster_size = int(min_cluster_size)
        self._pending: Dict[Tuple[str, str], Dict[str, Signal]] = defaultdict(dict)

    def process_signals(self, signals: Iterable[Signal]) -> List[Signal]:
        """Process ``signals`` and return newly confirmed cluster signals."""

        clustered: List[Signal] = []
        for signal in signals:
            symbol = signal.symbol.upper()
            key = (symbol, signal.side)
            bucket = self._pending[key]
            bucket[signal.strategy.upper()] = signal

            if len(bucket) < self.min_cluster_size:
                continue

            strategies = sorted(bucket.keys())
            contributing = [bucket[strategy] for strategy in strategies]
            confidence = sum(s.confidence for s in contributing) / len(contributing)

            metadata = {
                "cluster_size": len(strategies),
                "strategies": strategies,
                "source_signals": [
                    {
                        "strategy": s.strategy.upper(),
                        "confidence": s.confidence,
                        "metadata": dict(s.metadata),
                    }
                    for s in contributing
                ],
            }

            cluster_strategy = f"CLUSTER:{'-'.join(strategies)}"
            clustered.append(
                Signal(
                    symbol=symbol,
                    side=signal.side,
                    strategy=cluster_strategy,
                    confidence=confidence,
                    metadata=metadata,
                )
            )

            # Remove consumed signals so a new cluster requires fresh confirmations.
            self._pending.pop(key, None)

        return clustered

    def reset(self) -> None:
        """Clear pending confirmations (primarily useful for tests)."""

        self._pending.clear()


__all__ = ["ClusterEngine"]
