"""Trading bot package for the MyMarket demo application."""

from .config import AppConfig, PathsConfig, StrategyConfig, WebSocketConfig, ensure_data_paths, load_config
from .trading_bot import (
    Candle,
    DataManager,
    HistoricalBootstrapper,
    PriceProcessor,
    Signal,
    StrategyManager,
    Trade,
    TradingBot,
    TradingBotService,
    WebSocketManager,
)

__all__ = [
    "AppConfig",
    "PathsConfig",
    "StrategyConfig",
    "WebSocketConfig",
    "ensure_data_paths",
    "load_config",
    "Candle",
    "DataManager",
    "HistoricalBootstrapper",
    "PriceProcessor",
    "Signal",
    "StrategyManager",
    "Trade",
    "TradingBot",
    "TradingBotService",
    "WebSocketManager",
]
