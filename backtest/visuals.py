"""Plotting helpers for backtest reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_logger = logging.getLogger(__name__)

Number = float | int

_matplotlib_modules: tuple[object, object] | None = None


def _import_matplotlib(context: str) -> tuple[object, object] | None:
    """Import matplotlib modules using a headless backend."""

    global _matplotlib_modules
    if _matplotlib_modules is not None:
        return _matplotlib_modules

    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        _logger.warning("Unable to import matplotlib for %s plot: %s", context, exc)
        return None

    _matplotlib_modules = (mdates, plt)
    return _matplotlib_modules


def _resolve_point(
    point: Mapping[str, object] | Sequence[object]
) -> tuple[object | None, float]:
    """Return ``(time, value)`` for ``point`` regardless of structure."""

    if isinstance(point, Mapping):
        return point.get("time"), float(point.get("equity", 0.0))
    if len(point) >= 2:
        return point[0], float(point[1])
    return None, float(point[0]) if point else (None, 0.0)


def plot_equity_curve(
    equity_curve: Sequence[Mapping[str, object]]
    | Sequence[Sequence[object]]
    | Sequence[Number],
    output_path: str | Path,
) -> Path | None:
    """Generate an equity curve plot."""

    modules = _import_matplotlib("equity")
    if modules is None:
        return None
    mdates, plt = modules

    if not equity_curve:
        return None

    times: list[object | None] = []
    values: list[float] = []

    first = equity_curve[0]
    if isinstance(first, (Mapping, Sequence)) and not isinstance(first, (int, float)):
        for point in equity_curve:
            time, value = _resolve_point(point)  # type: ignore[arg-type]
            times.append(time)
            values.append(float(value))
    else:
        values = [float(value) for value in equity_curve]  # type: ignore[arg-type]

    fig, ax = plt.subplots(figsize=(10, 5))
    if any(times):
        ax.plot(times, values, label="Equity", color="tab:blue")
        ax.set_xlabel("Time")
        fig.autofmt_xdate()
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.plot(range(len(values)), values, label="Equity", color="tab:blue")
        ax.set_xlabel("Observation")

    ax.set_ylabel("Balance")
    ax.set_title("Equity curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_price_with_trades(
    price_series: Sequence[Mapping[str, object]],
    trades: Iterable[Mapping[str, object]],
    output_path: str | Path,
) -> Path | None:
    """Plot the price series with entry/exit markers."""

    modules = _import_matplotlib("price")
    if modules is None:
        return None
    mdates, plt = modules

    if not price_series:
        return None

    times = [point.get("time") for point in price_series]
    closes = [float(point.get("close", 0.0)) for point in price_series]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, closes, label="Close", color="tab:gray", linewidth=1.0)

    entries_x: list[object] = []
    entries_y: list[float] = []
    exits_x: list[object] = []
    exits_y: list[float] = []
    colors: list[str] = []

    for trade in trades:
        entry_time = trade.get("entry_time")
        entry_price = trade.get("entry_price")
        exit_time = trade.get("exit_time")
        exit_price = trade.get("exit_price")
        side = trade.get("side", "LONG")
        if entry_time is not None and entry_price is not None:
            entries_x.append(entry_time)
            entries_y.append(float(entry_price))
            colors.append("tab:green" if side == "LONG" else "tab:red")
        if exit_time is not None and exit_price is not None:
            exits_x.append(exit_time)
            exits_y.append(float(exit_price))

    if entries_x:
        ax.scatter(entries_x, entries_y, marker="^", color=colors, label="Entries")
    if exits_x:
        ax.scatter(exits_x, exits_y, marker="x", color="tab:orange", label="Exits")

    ax.set_title("Price action with trades")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


__all__ = ["plot_equity_curve", "plot_price_with_trades"]
