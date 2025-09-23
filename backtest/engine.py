"""Backtesting engine with trailing stop support and reporting helpers."""

from __future__ import annotations

import csv
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from binance_client import Kline
from module_base import ModuleBase, Signal

from .metrics import summarize_performance
from .visuals import plot_equity_curve, plot_price_with_trades

_logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Represents a single trade during a backtest run."""

    trade_id: str
    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    entry_index: int
    entry_equity: float
    trailing_percent: float
    enable_trailing: bool
    commission_pct: float
    quantity: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)
    trailing_stop: Optional[float] = None
    initial_stop: Optional[float] = None
    high_watermark: float = field(init=False)
    low_watermark: float = field(init=False)
    max_profit_pct: float = 0.0
    current_profit_pct: float = 0.0
    current_price: float = field(init=False)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_index: Optional[int] = None
    exit_reason: Optional[str] = None
    profit_pct: Optional[float] = None
    profit_value: Optional[float] = None
    duration_bars: Optional[int] = None
    is_active: bool = True

    def __post_init__(self) -> None:
        self.high_watermark = self.entry_price
        self.low_watermark = self.entry_price
        self.current_price = self.entry_price
        if self.enable_trailing and self.trailing_percent > 0:
            if self.side == "LONG":
                self.initial_stop = self.entry_price * (1 - self.trailing_percent / 100.0)
            else:
                self.initial_stop = self.entry_price * (1 + self.trailing_percent / 100.0)
            self.trailing_stop = self.initial_stop
        else:
            self.initial_stop = None
            self.trailing_stop = None

    def apply_price(
        self,
        price: float,
        timestamp: datetime,
        *,
        bar_index: int,
        enable_trailing: bool,
    ) -> bool:
        """Update trailing stop and optionally close the trade."""

        if not self.is_active:
            return False

        self.current_price = price
        profit_pct = self._calculate_profit_pct(price)
        self.current_profit_pct = profit_pct
        if profit_pct > self.max_profit_pct:
            self.max_profit_pct = profit_pct

        if not (enable_trailing and self.enable_trailing and self.trailing_percent > 0):
            return False

        if self.side == "LONG":
            if price > self.high_watermark:
                self.high_watermark = price
                new_stop = price * (1 - self.trailing_percent / 100.0)
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
            if self.trailing_stop is not None and price <= self.trailing_stop:
                self.close(
                    price,
                    timestamp,
                    reason="trailing_stop",
                    bar_index=bar_index,
                )
                return True
        else:
            if price < self.low_watermark:
                self.low_watermark = price
                new_stop = price * (1 + self.trailing_percent / 100.0)
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
            if self.trailing_stop is not None and price >= self.trailing_stop:
                self.close(
                    price,
                    timestamp,
                    reason="trailing_stop",
                    bar_index=bar_index,
                )
                return True
        return False

    def close(
        self,
        price: float,
        timestamp: datetime,
        *,
        reason: str,
        bar_index: Optional[int] = None,
    ) -> None:
        """Finalize the trade with the provided ``price``."""

        if not self.is_active:
            return

        gross_profit_pct = self._calculate_profit_pct(price)
        net_profit_pct = gross_profit_pct - self.commission_pct

        self.exit_price = price
        self.exit_time = timestamp
        self.exit_index = bar_index
        self.exit_reason = reason
        self.profit_pct = net_profit_pct
        self.current_profit_pct = net_profit_pct
        self.profit_value = self.entry_equity * (net_profit_pct / 100.0)
        self.duration_bars = (bar_index - self.entry_index) if bar_index is not None else None
        self.trailing_stop = price
        self.is_active = False
        self.current_price = price

    def to_dict(self) -> Dict[str, object]:
        """Serialize the trade to a dictionary."""

        return {
            "id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": float(self.entry_price),
            "entry_time": self.entry_time.isoformat(),
            "entry_index": self.entry_index,
            "exit_price": float(self.exit_price) if self.exit_price is not None else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_index": self.exit_index,
            "exit_reason": self.exit_reason,
            "profit_pct": float(self.profit_pct) if self.profit_pct is not None else None,
            "profit_value": float(self.profit_value) if self.profit_value is not None else None,
            "duration_bars": self.duration_bars,
            "trailing_percent": float(self.trailing_percent),
            "initial_stop": float(self.initial_stop) if self.initial_stop is not None else None,
            "trailing_stop": float(self.trailing_stop) if self.trailing_stop is not None else None,
            "max_profit_pct": float(self.max_profit_pct),
            "current_profit_pct": float(self.current_profit_pct),
            "confidence": float(self.confidence),
            "metadata": self.metadata,
        }

    def _calculate_profit_pct(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry_price) / self.entry_price * 100.0
        return (self.entry_price - price) / self.entry_price * 100.0


@dataclass
class BacktestResult:
    """Container returned after a backtest completes."""

    symbol: str
    strategy: str
    metrics: Dict[str, float | int]
    trades: List[Dict[str, object]]
    equity_curve: List[Dict[str, object]]
    report_directory: Path
    artifacts: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "metrics": self.metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "report_directory": str(self.report_directory),
            "artifacts": dict(self.artifacts),
        }


class Backtester:
    """Iterate over historical candles and evaluate a strategy."""

    def __init__(
        self,
        strategy: ModuleBase,
        candles: Sequence[Kline],
        *,
        symbol: str,
        trailing_percent: float = 2.0,
        enable_trailing: bool = True,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.0,
        report_directory: str | Path | None = None,
    ) -> None:
        if not candles:
            raise ValueError("historical candle series is empty")

        self.strategy = strategy
        self.symbol = symbol.upper()
        self.candles: List[Kline] = list(candles)
        self.trailing_percent = float(trailing_percent)
        self.enable_trailing = enable_trailing
        self.initial_capital = float(initial_capital)
        self.commission_pct = float(commission_pct)
        self.report_directory = Path(report_directory or "backtest_reports")
        self.report_directory.mkdir(parents=True, exist_ok=True)

        self._price_series: List[Dict[str, object]] = [
            {
                "time": _as_datetime(candle.close_time),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in self.candles
        ]

        self._completed_trades: List[BacktestTrade] = []
        self._equity_curve: List[Dict[str, object]] = []

    @property
    def price_series(self) -> List[Dict[str, object]]:
        return list(self._price_series)

    @property
    def equity_curve(self) -> List[Dict[str, object]]:
        return list(self._equity_curve)

    def run(self) -> BacktestResult:
        """Execute the backtest and generate a performance report."""

        if len(self.candles) < self.strategy.minimum_bars:
            raise ValueError(
                "not enough candles for strategy "
                f"(required {self.strategy.minimum_bars}, have {len(self.candles)})"
            )

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_directory = self.report_directory / f"{self.strategy.abbreviation}_{self.symbol}_{timestamp}"
        run_directory.mkdir(parents=True, exist_ok=True)

        _logger.info(
            "Starting backtest for %s (%s) with %d candles",
            self.symbol,
            self.strategy.abbreviation,
            len(self.candles),
        )

        equity = self.initial_capital
        current_trade: Optional[BacktestTrade] = None
        self._completed_trades.clear()
        self._equity_curve = [
            {
                "time": _as_datetime(self.candles[0].open_time),
                "equity": equity,
            }
        ]

        for index, candle in enumerate(self.candles):
            candle_time = _as_datetime(candle.close_time)
            price = candle.close

            if current_trade and current_trade.is_active:
                closed = current_trade.apply_price(
                    price,
                    candle_time,
                    bar_index=index,
                    enable_trailing=self.enable_trailing,
                )
                if closed:
                    self._finalise_trade(current_trade)
                    equity = current_trade.entry_equity * (1 + (current_trade.profit_pct or 0.0) / 100.0)
                    self._equity_curve.append({"time": candle_time, "equity": equity})
                    current_trade = None
                    continue
                floating_equity = current_trade.entry_equity * (
                    1 + current_trade.current_profit_pct / 100.0
                )
                self._equity_curve.append({"time": candle_time, "equity": floating_equity})
            else:
                self._equity_curve.append({"time": candle_time, "equity": equity})

            if index + 1 < self.strategy.minimum_bars:
                continue

            try:
                signals = list(self.strategy.process(self.symbol, self.candles[: index + 1]))
            except Exception:  # pragma: no cover - defensive
                _logger.exception("strategy %s failed at index %d", self.strategy.abbreviation, index)
                continue
            if not signals:
                continue

            signal = signals[0]
            if current_trade and current_trade.is_active:
                if signal.side != current_trade.side:
                    current_trade.close(
                        price,
                        candle_time,
                        reason="signal_flip",
                        bar_index=index,
                    )
                    self._finalise_trade(current_trade)
                    equity = current_trade.entry_equity * (
                        1 + (current_trade.profit_pct or 0.0) / 100.0
                    )
                    self._equity_curve.append({"time": candle_time, "equity": equity})
                    current_trade = None
                continue

            current_trade = self._open_trade(signal, price, candle_time, index, equity)

        if current_trade and current_trade.is_active:
            last_candle = self.candles[-1]
            close_time = _as_datetime(last_candle.close_time)
            current_trade.close(
                last_candle.close,
                close_time,
                reason="end_of_data",
                bar_index=len(self.candles) - 1,
            )
            self._finalise_trade(current_trade)
            equity = current_trade.entry_equity * (1 + (current_trade.profit_pct or 0.0) / 100.0)
            self._equity_curve[-1] = {"time": close_time, "equity": equity}
            current_trade = None

        trades_payload = [trade.to_dict() for trade in self._completed_trades]
        equity_payload = [
            {
                "time": point["time"].isoformat() if isinstance(point.get("time"), datetime) else point.get("time"),
                "equity": float(point.get("equity", 0.0)),
            }
            for point in self._equity_curve
        ]

        metrics = summarize_performance(
            trades_payload,
            equity_payload,
            initial_capital=self.initial_capital,
        )

        artifacts = self._save_report(
            run_directory,
            trades_payload,
            equity_payload,
            metrics,
        )

        _logger.info(
            "Backtest finished: %d trades, %.2f%% return",
            metrics.get("num_trades", 0),
            metrics.get("total_return_pct", 0.0),
        )

        return BacktestResult(
            symbol=self.symbol,
            strategy=self.strategy.abbreviation,
            metrics=metrics,
            trades=trades_payload,
            equity_curve=equity_payload,
            report_directory=run_directory,
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------
    def _open_trade(
        self,
        signal: Signal,
        price: float,
        timestamp: datetime,
        index: int,
        equity: float,
    ) -> BacktestTrade:
        trade = BacktestTrade(
            trade_id=str(uuid.uuid4()),
            symbol=self.symbol,
            side=signal.side,
            entry_price=price,
            entry_time=timestamp,
            entry_index=index,
            entry_equity=equity,
            trailing_percent=self.trailing_percent,
            enable_trailing=self.enable_trailing,
            commission_pct=self.commission_pct,
            quantity=1.0,
            confidence=getattr(signal, "confidence", 1.0) or 1.0,
            metadata=dict(getattr(signal, "metadata", {}) or {}),
        )
        _logger.debug(
            "Opening %s trade at %.2f (confidence %.2f)",
            trade.side,
            trade.entry_price,
            trade.confidence,
        )
        return trade

    # ------------------------------------------------------------------
    def _finalise_trade(self, trade: BacktestTrade) -> None:
        if trade not in self._completed_trades:
            self._completed_trades.append(trade)

    # ------------------------------------------------------------------
    def _save_report(
        self,
        directory: Path,
        trades: List[Dict[str, object]],
        equity_curve: List[Dict[str, object]],
        metrics: Dict[str, float | int],
    ) -> Dict[str, str]:
        artifacts: Dict[str, str] = {}

        trades_path = directory / "trades.json"
        with trades_path.open("w", encoding="utf8") as handle:
            json.dump(trades, handle, indent=2)
        artifacts["trades_json"] = str(trades_path)

        csv_path = directory / "trades.csv"
        if trades:
            fieldnames = sorted({key for trade in trades for key in trade.keys()})
            with csv_path.open("w", encoding="utf8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trades)
            artifacts["trades_csv"] = str(csv_path)

        equity_path = directory / "equity_curve.json"
        with equity_path.open("w", encoding="utf8") as handle:
            json.dump(equity_curve, handle, indent=2)
        artifacts["equity_json"] = str(equity_path)

        metrics_path = directory / "metrics.json"
        with metrics_path.open("w", encoding="utf8") as handle:
            json.dump(metrics, handle, indent=2)
        artifacts["metrics_json"] = str(metrics_path)

        equity_plot = plot_equity_curve(self._equity_curve, directory / "equity_curve.png")
        if equity_plot:
            artifacts["equity_plot"] = str(equity_plot)

        plot_trades = [
            {
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "exit_time": trade.exit_time,
                "exit_price": trade.exit_price,
                "side": trade.side,
            }
            for trade in self._completed_trades
        ]
        price_plot = plot_price_with_trades(
            self._price_series,
            plot_trades,
            directory / "price_trades.png",
        )
        if price_plot:
            artifacts["price_plot"] = str(price_plot)

        return artifacts


def load_klines_from_csv(path: str | Path) -> List[Kline]:
    """Load klines from ``path`` and return them as :class:`Kline` objects."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)

    with resolved.open("r", encoding="utf8", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect)
        rows = list(reader)

    if not rows:
        return []

    header = [cell.strip().lower() for cell in rows[0]]
    required = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    start_index = 0
    if all(key in header for key in required):
        index_map = {key: header.index(key) for key in required}
        start_index = 1
    else:
        index_map = {key: i for i, key in enumerate(required)}

    candles: List[Kline] = []
    for row in rows[start_index:]:
        if len(row) < 7:
            continue
        payload = [row[index_map[key]] for key in required]
        try:
            candles.append(Kline.from_rest(payload))
        except Exception:  # pragma: no cover - defensive
            _logger.exception("Failed to parse row in %s", resolved)
            continue

    candles.sort(key=lambda candle: candle.open_time)
    return candles


def _as_datetime(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)


__all__ = [
    "Backtester",
    "BacktestTrade",
    "BacktestResult",
    "load_klines_from_csv",
]
