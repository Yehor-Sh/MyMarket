# MyMarket

A lightweight research environment for experimenting with automated trading
strategies on Binance spot markets.  The project is intentionally simple and
focuses on presenting a clear separation of responsibilities between market data
collection, strategy evaluation and trade management.

## Components

- **`binance_client.py`** – REST-based data client that keeps an in-memory cache
  of ticker prices and klines while filtering liquid pairs.
- **`modules/`** – Strategy modules that analyse candlestick data and emit
  trading signals without worrying about order execution.
- **`module_worker.py`** – Launches modules on independent threads and funnels
  signals to the orchestrator through a queue.
- **`orchestrator.py`** – Flask + Socket.IO service that manages trades,
  applies trailing stops and streams updates to the UI over WebSocket.
- **`client.html`** – Lightweight dashboard that subscribes to WebSocket
  updates to display open and closed positions in real time.

## Running locally

```bash
pip install -r requirements.txt
python orchestrator.py
```

The web interface becomes available at <http://localhost:8080>.  The
orchestrator automatically starts the Binance client and all strategy workers.
Networking failures are tolerated – when the Binance API is unreachable the
system continues to operate with the latest cached data.
