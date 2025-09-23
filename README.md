# MyMarket

A lightweight research environment for experimenting with automated trading
strategies on Binance spot markets.  The project is intentionally simple and
focuses on presenting a clear separation of responsibilities between market data
collection, strategy evaluation and trade management.

## Components

- **`binance_client.py`** – REST/WebSocket data client that keeps an in-memory
  cache of ticker prices and klines while filtering liquid pairs.
- **`modules/`** – Strategy modules that analyse candlestick data and emit
  trading signals without worrying about order execution.
- **`module_worker.py`** – Launches modules on independent threads and funnels
  signals to the orchestrator through a queue.
- **`orchestrator.py`** – Flask + Socket.IO service that manages trades,
  applies trailing stops and streams updates to the UI over WebSocket.
- **`client.html`** – Lightweight dashboard that subscribes to WebSocket
  updates to display open and closed positions in real time.
- **`backtest/`** – Offline backtesting engine with reporting utilities
  (metrics and visualisations).

## Running locally

```bash
pip install -r requirements.txt
python orchestrator.py
```

The web interface becomes available at <http://localhost:8080>.  The
orchestrator automatically starts the Binance client and all strategy workers.
Networking failures are tolerated – when the Binance API is unreachable the
system continues to operate with the latest cached data.

## Backtesting mode

The project ships with an offline backtester that replays historical data bar
by bar.  Configure the behaviour through `config.py`:

```python
CONFIG = {
    "mode": "backtest",  # switch between "live" and "backtest"
    "backtest": {
        "symbol": "BTCUSDT",
        "strategy": "ENG",  # strategy abbreviation
        "csv_path": "data/BTCUSDT.csv",  # optional CSV with klines
        "trailing_percent": 2.0,
        "initial_capital": 10_000.0,
    },
}
```

When `mode` is set to `backtest`, running `python orchestrator.py` executes the
selected strategy on the provided historical data and produces a report in the
`backtest_reports/` directory (trades, metrics and PNG charts).
