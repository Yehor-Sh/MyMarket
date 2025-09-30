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

## Running locally

```bash
pip install -r requirements.txt
python orchestrator.py
```

The web interface becomes available at <http://localhost:8080>.  The
orchestrator automatically starts the Binance client and all strategy workers.
Networking failures are tolerated – when the Binance API is unreachable the
system continues to operate with the latest cached data.  If the client is
started without a previously cached liquidity snapshot it automatically falls
back to a small offline list of symbols (``BTCUSDT``/``ETHUSDT`` by default).
The fallback list can be overridden by passing ``offline_pairs`` to
``BinanceClient`` or wiring the parameter through your own configuration.

## Manual QA

- **Closed trade push event:** Open the dashboard, ensure at least one active
  trade is visible, and emit only a ``trade_closed`` Socket.IO event for that
  trade (without broadcasting a full ``trades`` snapshot). Confirm that the
  position disappears from the «Активные сделки» table, reappears under
  «Закрытые сделки» and that aggregate statistics update instantly.
