"""Performance metric helpers for the backtesting engine."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence

Number = float | int


def _extract_equity_values(
    equity_curve: Sequence[Number] | Sequence[Mapping[str, Number]]
) -> list[float]:
    """Return equity values as a list of floats."""

    if not equity_curve:
        return []
    first = equity_curve[0]
    if isinstance(first, Mapping):
        return [float(point.get("equity", 0.0)) for point in equity_curve]
    return [float(value) for value in equity_curve]


def compute_max_drawdown(
    equity_curve: Sequence[Number] | Sequence[Mapping[str, Number]]
) -> float:
    """Return the maximal drawdown in percent for ``equity_curve``."""

    values = _extract_equity_values(equity_curve)
    if not values:
        return 0.0

    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        if value > peak:
            peak = value
            continue
        if peak <= 0:
            continue
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown * 100.0


def compute_sharpe_ratio(
    returns: Iterable[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Return the annualised Sharpe ratio for ``returns``."""

    returns = list(returns)
    if len(returns) < 2:
        return 0.0

    average_return = mean(returns)
    adjusted_return = average_return - (risk_free_rate / max(1, periods_per_year))
    variance = pstdev(returns)
    if math.isclose(variance, 0.0):
        return 0.0
    return (adjusted_return / variance) * math.sqrt(periods_per_year)


def summarize_performance(
    trades: Sequence[Mapping[str, float]] | Sequence[object],
    equity_curve: Sequence[Number] | Sequence[Mapping[str, Number]],
    *,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Compute headline statistics for a backtest run."""

    equity_values = _extract_equity_values(equity_curve)
    final_equity = equity_values[-1] if equity_values else initial_capital

    total_return_pct = 0.0
    if initial_capital > 0:
        total_return_pct = ((final_equity / initial_capital) - 1.0) * 100.0

    profit_percentages: list[float] = []
    for trade in trades:
        value = None
        if isinstance(trade, Mapping):
            value = trade.get("profit_pct")
        if value is None:
            value = getattr(trade, "profit_pct", None)
        if value is None:
            continue
        profit_percentages.append(float(value))

    num_trades = len(profit_percentages)
    wins = sum(1 for value in profit_percentages if value > 0)
    win_rate_pct = (wins / num_trades * 100.0) if num_trades else 0.0
    average_trade_pct = mean(profit_percentages) if profit_percentages else 0.0

    max_drawdown_pct = compute_max_drawdown(equity_values)
    sharpe_ratio = compute_sharpe_ratio(
        (value / 100.0 for value in profit_percentages),
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    return {
        "initial_capital": float(initial_capital),
        "final_equity": float(final_equity),
        "total_return_pct": float(total_return_pct),
        "num_trades": int(num_trades),
        "win_rate_pct": float(win_rate_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "average_trade_pct": float(average_trade_pct),
        "sharpe_ratio": float(sharpe_ratio),
    }


__all__ = [
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "summarize_performance",
]
