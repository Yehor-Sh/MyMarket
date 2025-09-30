from __future__ import annotations

import pytest

from module_base import Signal
from modules.cluster_engine import ClusterEngine


def _signal(strategy: str, confidence: float = 1.0, symbol: str = "BTCUSDT", side: str = "LONG") -> Signal:
    return Signal(
        symbol=symbol,
        side=side,
        strategy=strategy,
        confidence=confidence,
        metadata={"source": strategy},
    )


def test_cluster_emitted_only_after_threshold() -> None:
    engine = ClusterEngine(min_cluster_size=3)

    first_batch = engine.process_signals([
        _signal("AAA", 0.8),
        _signal("BBB", 0.6),
    ])
    assert first_batch == []

    second_batch = engine.process_signals([_signal("CCC", 0.4)])
    assert len(second_batch) == 1

    clustered = second_batch[0]
    assert clustered.strategy == "CLUSTER:AAA-BBB-CCC"
    assert clustered.confidence == pytest.approx((0.8 + 0.6 + 0.4) / 3)
    assert clustered.metadata["cluster_size"] == 3
    assert clustered.metadata["strategies"] == ["AAA", "BBB", "CCC"]


def test_duplicate_strategy_replaced_not_counted() -> None:
    engine = ClusterEngine(min_cluster_size=3)

    engine.process_signals([_signal("AAA", 0.2, symbol="ETHUSDT", side="SHORT")])
    engine.process_signals([_signal("AAA", 0.9, symbol="ETHUSDT", side="SHORT")])
    result = engine.process_signals([
        _signal("BBB", 0.3, symbol="ETHUSDT", side="SHORT"),
        _signal("CCC", 0.5, symbol="ETHUSDT", side="SHORT"),
    ])

    assert len(result) == 1
    clustered = result[0]
    assert clustered.strategy == "CLUSTER:AAA-BBB-CCC"
    assert clustered.confidence == pytest.approx((0.9 + 0.3 + 0.5) / 3)


def test_pending_cleared_after_cluster() -> None:
    engine = ClusterEngine(min_cluster_size=2)

    initial = engine.process_signals([
        _signal("AAA", 0.7),
        _signal("BBB", 0.4),
    ])
    assert len(initial) == 1

    follow_up = engine.process_signals([_signal("AAA", 0.9)])
    assert follow_up == []


