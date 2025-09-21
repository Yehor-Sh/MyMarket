"""Flask + Socket.IO application serving the demo trading bot."""
from __future__ import annotations

from typing import Tuple

from flask import Flask, jsonify
from flask_socketio import SocketIO

from .config import AppConfig, load_config
from .trading_bot import TradingBot, TradingBotService


def create_app(config: AppConfig | None = None) -> Tuple[Flask, SocketIO, TradingBotService]:
    config = config or load_config()
    app = Flask(__name__, static_folder="static", static_url_path="")
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    bot = TradingBot(config)
    service = TradingBotService(bot)

    def emit_candle(payload):
        socketio.emit("candle", payload, namespace="/quotes")

    def emit_trade(payload):
        socketio.emit("trade", payload, namespace="/quotes")

    def emit_signal(payload):
        socketio.emit("signal", payload, namespace="/quotes")

    bot.register_listener("candle", emit_candle)
    bot.register_listener("trade", emit_trade)
    bot.register_listener("signal", emit_signal)

    @app.before_first_request
    def _start_bot() -> None:  # pragma: no cover - integration path
        service.start()

    @app.route("/api/candles")
    def api_candles():
        return jsonify(bot.get_candles())

    @app.route("/api/trades")
    def api_trades():
        return jsonify(bot.get_trades())

    @app.route("/")
    def index():  # pragma: no cover - handled by Flask machinery
        return app.send_static_file("index.html")

    return app, socketio, service


def main() -> None:  # pragma: no cover - manual execution entrypoint
    app, socketio, service = create_app()
    try:
        socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
    finally:
        service.stop()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
