"""Configuration management for the demo trading bot."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Optional


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass
class WebSocketConfig:
    """Parameters for the WebSocket price stream."""

    stream_url: str = os.environ.get("MARKET_STREAM_URL", "wss://stream.binance.com:9443/ws")
    reconnect_delay: float = float(os.environ.get("MARKET_RECONNECT_DELAY", "2.0"))
    price_interval_seconds: int = int(os.environ.get("MARKET_PRICE_INTERVAL", "60"))
    symbol: str = os.environ.get("MARKET_SYMBOL", "BTCUSDT")


@dataclass
class StrategyConfig:
    """Parameters that control the demo trading strategy."""

    lookback: int = int(os.environ.get("STRATEGY_LOOKBACK", "8"))
    fast_ma: int = int(os.environ.get("STRATEGY_FAST_MA", "3"))
    slow_ma: int = int(os.environ.get("STRATEGY_SLOW_MA", "6"))
    trailing_stop_pct: float = float(os.environ.get("STRATEGY_TRAILING_PCT", "0.01"))
    min_volume: float = float(os.environ.get("STRATEGY_MIN_VOLUME", "0.0"))


@dataclass
class PathsConfig:
    """Filesystem locations for persistent bot data."""

    data_dir: Path = DEFAULT_DATA_DIR
    candles_file: Path = data_dir / "candles.json"
    trades_file: Path = data_dir / "trades.json"

    @classmethod
    def from_env(cls) -> "PathsConfig":
        data_dir = Path(os.environ.get("BOT_DATA_DIR", str(DEFAULT_DATA_DIR)))
        candles_path = Path(os.environ.get("BOT_CANDLES_FILE", str(data_dir / "candles.json")))
        trades_path = Path(os.environ.get("BOT_TRADES_FILE", str(data_dir / "trades.json")))
        return cls(data_dir=data_dir, candles_file=candles_path, trades_file=trades_path)


@dataclass
class AppConfig:
    """Top level application configuration container."""

    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig.from_env)
    bootstrap_url: Optional[str] = os.environ.get("MARKET_REST_URL")


def ensure_data_paths(paths: PathsConfig) -> None:
    """Ensure that the directory structure for JSON files exists."""

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    if not paths.candles_file.exists():
        payload = {"symbol": "DEMO", "interval": "1m", "candles": []}
        paths.candles_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not paths.trades_file.exists():
        payload = {"active": [], "closed": []}
        paths.trades_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_config() -> AppConfig:
    """Load configuration using environment variables and defaults."""

    config = AppConfig()
    ensure_data_paths(config.paths)
    return config


__all__ = [
    "AppConfig",
    "StrategyConfig",
    "WebSocketConfig",
    "PathsConfig",
    "load_config",
    "ensure_data_paths",
]
