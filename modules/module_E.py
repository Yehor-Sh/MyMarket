#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
import uuid
from typing import Dict, List, Any, Optional

import httpx
import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8080")
SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "dev-token")

BARS_FILE_PATH = os.environ.get("BARS_FILE_PATH", "./bars_1m.json")
INTERVAL_STRING = os.environ.get("INTERVAL_STRING", "1m")
MODULE_NAME = os.environ.get("MODULE_NAME", "module_E_momentum_roc")
MODULE_VERSION = os.environ.get("MODULE_VERSION", "1.1.0")

ROC_PERIOD = int(os.environ.get("ROC_PERIOD", "10"))
POSITIVE_THRESHOLD = float(os.environ.get("POSITIVE_THRESHOLD", "0.006"))
NEGATIVE_THRESHOLD = float(os.environ.get("NEGATIVE_THRESHOLD", "-0.006"))
MIN_ABS_ROC = float(os.environ.get("MIN_ABS_ROC", "0.004"))

MIN_PRICE = float(os.environ.get("MIN_PRICE", "0.3"))
MAX_PRICE = float(os.environ.get("MAX_PRICE", "5000"))

DEFAULT_QTY = float(os.environ.get("DEFAULT_QTY", "1.0"))
TRAILING_TYPE = os.environ.get("TRAILING_TYPE", "percent")
TRAILING_VALUE = float(os.environ.get("TRAILING_VALUE", "0.01"))
SEND_UPDATES_IF_NO_SIGNALS = os.environ.get("SEND_UPDATES_IF_NO_SIGNALS", "0") == "1"

COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "300"))
VOL_MULTIPLIER = float(os.environ.get("VOL_MULTIPLIER", "1.3"))
SMA_SHORT = int(os.environ.get("SMA_SHORT", "20"))
SMA_LONG = int(os.environ.get("SMA_LONG", "50"))

ZSCORE_ENABLE = os.environ.get("ZSCORE_ENABLE", "1") == "1"
ZSCORE_WINDOW = int(os.environ.get("ZSCORE_WINDOW", "60"))
ZSCORE_MIN_SIGMA = float(os.environ.get("ZSCORE_MIN_SIGMA", "0.002"))
ZSCORE_POS_THRESH = float(os.environ.get("ZSCORE_POS_THRESH", "1.2"))
ZSCORE_NEG_THRESH = float(os.environ.get("ZSCORE_NEG_THRESH", "1.2"))

REQUEST_TIMEOUT = 10.0
MODULE_HTTP_TIMEOUT = float(os.environ.get("MODULE_HTTP_TIMEOUT", "5.0"))


def load_json(path: str, default_value: Any) -> Any:
    try:
        if not os.path.exists(path):
            return default_value
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value


def load_snapshot(path: str) -> Dict[str, Any]:
    return load_json(path, {"symbols": {}, "meta": {}})


def to_closed_numeric(bars: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for b in bars:
        if not b.get("is_closed"):
            continue
        try:
            out.append({
                "open_time": float(b["open_time"]),
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
                "volume": float(b["volume"]),
            })
        except Exception:
            continue
    return out


def median(values: List[float]) -> Optional[float]:
    n = len(values)
    if n == 0:
        return None
    s = sorted(values)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return float((s[m - 1] + s[m]) / 2.0)


def sma(values: List[float], n: int) -> Optional[float]:
    if n <= 0 or len(values) < n:
        return None
    return float(sum(values[-n:]) / n)


def compute_roc_series(closes: List[float], period: int) -> List[float]:
    out: List[float] = []
    for i in range(period, len(closes)):
        ref = closes[i - period]
        if ref <= 0:
            out.append(0.0)
        else:
            out.append((closes[i] - ref) / ref)
    return out


def stddev(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) * (x - mean) for x in values) / (n - 1)
    return var ** 0.5


def detect_signals(snapshot: Dict[str, Any], mod_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    symbols_map: Dict[str, List[Dict[str, Any]]] = snapshot.get("symbols", {})
    now_ms = int(time.time() * 1000)
    now_sec = int(time.time())
    need = max(ROC_PERIOD + 1, SMA_LONG + 1, ZSCORE_WINDOW + ROC_PERIOD if ZSCORE_ENABLE else 0)

    for symbol, bars in symbols_map.items():
        arr = to_closed_numeric(bars)
        if len(arr) < need:
            continue

        closes = [x["close"] for x in arr]
        vols = [x["volume"] for x in arr]
        last_close = float(closes[-1])
        ref_price = float(closes[-(ROC_PERIOD + 1)])

        if ref_price <= 0.0:
            continue
        if not (MIN_PRICE <= last_close <= MAX_PRICE):
            continue

        roc = (last_close - ref_price) / ref_price
        if abs(roc) < MIN_ABS_ROC:
            continue

        v_med = median(vols[-(SMA_SHORT + 1):-1]) or 0.0
        last_vol = float(vols[-1])
        if v_med <= 0.0 or last_vol < v_med * VOL_MULTIPLIER:
            continue

        s_short = sma(closes, SMA_SHORT)
        s_long = sma(closes, SMA_LONG)
        if s_short is None or s_long is None:
            continue

        direction = None
        if roc >= POSITIVE_THRESHOLD and last_close >= s_short >= s_long:
            direction = "long"
        elif roc <= NEGATIVE_THRESHOLD and last_close <= s_short <= s_long:
            direction = "short"
        else:
            continue

        if ZSCORE_ENABLE:
            roc_series = compute_roc_series(closes[-(ZSCORE_WINDOW + ROC_PERIOD):], ROC_PERIOD)
            sigma = stddev(roc_series[-ZSCORE_WINDOW:]) if len(roc_series) >= ZSCORE_WINDOW else 0.0
            if sigma < ZSCORE_MIN_SIGMA:
                continue
            z = roc / sigma if sigma > 0 else 0.0
            if direction == "long" and z < ZSCORE_POS_THRESH:
                continue
            if direction == "short" and abs(z) < ZSCORE_NEG_THRESH:
                continue

        ms_key = f"{MODULE_NAME}:{INTERVAL_STRING}:{symbol}:{direction}"
        last_ts = int(mod_state.get(ms_key, 0))
        if now_sec - last_ts < COOLDOWN_SEC:
            continue

        signals.append({
            "symbol": symbol,
            "direction": direction,
            "action": "enter",
            "price": last_close,
            "quantity": DEFAULT_QTY,
            "stop_loss": None,
            "trailing": {"type": TRAILING_TYPE, "value": TRAILING_VALUE},
            "time": now_ms,
            "meta": {
                "pattern": "roc_positive" if direction == "long" else "roc_negative",
                "roc": float(roc),
                "period": ROC_PERIOD,
                "v_med": float(v_med),
                "last_vol": float(last_vol),
                "sma_short": float(s_short),
                "sma_long": float(s_long)
            }
        })
        mod_state[ms_key] = now_sec

    return signals


def fetch_module_state() -> Dict[str, Any]:
    url = f"{SERVER_URL}/modules/state"
    try:
        response = requests.get(url, timeout=MODULE_HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {}
        modules = payload.get("modules")
        if not isinstance(modules, dict):
            return {}
        entry = modules.get(MODULE_NAME)
        if not isinstance(entry, dict):
            return {}
        state = entry.get("state")
        if isinstance(state, dict):
            return dict(state)
        return {k: v for k, v in entry.items() if k not in ("module", "updated_at")}
    except Exception as exc:
        print(f"[module E] state_fetch_failed: {exc}")
        return {}


def persist_module_state(mod_state: Dict[str, Any]) -> None:
    url = f"{SERVER_URL}/modules/state"
    headers = {"Authorization": f"Bearer {SERVER_TOKEN}"}
    payload = {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "interval": INTERVAL_STRING,
        "state": mod_state,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=MODULE_HTTP_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:
        print(f"[module E] state_persist_failed: {exc}")


def register_module(client: httpx.Client) -> str:
    r = client.post(
        f"{SERVER_URL}/modules/register",
        headers={"Authorization": f"Bearer {SERVER_TOKEN}"},
        json={
            "module_name": MODULE_NAME,
            "version": MODULE_VERSION,
            "capabilities": {"intervals": [INTERVAL_STRING]},
            "expires_in_sec": 180,
            "request_id": str(uuid.uuid4()),
        },
        timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    return r.json()["module_id"]


def report_signals(client: httpx.Client, module_id: str, signals: List[Dict[str, Any]]) -> None:
    payload = {
        "module_id": module_id,
        "interval": INTERVAL_STRING,
        "signals": signals,
        "request_id": str(uuid.uuid4()),
    }
    r = client.post(
        f"{SERVER_URL}/signals/report",
        headers={"Authorization": f"Bearer {SERVER_TOKEN}"},
        json=payload,
        timeout=REQUEST_TIMEOUT + 5
    )
    r.raise_for_status()
    print("[module E]", r.json())


def main() -> None:
    snapshot = load_snapshot(BARS_FILE_PATH)
    mod_state = fetch_module_state()
    signals = detect_signals(snapshot, mod_state)
    if not signals and not SEND_UPDATES_IF_NO_SIGNALS:
        print("[module E] no_signals")
        persist_module_state(mod_state)
        return
    with httpx.Client() as client:
        module_id = register_module(client)
        report_signals(client, module_id, signals)
    persist_module_state(mod_state)


if __name__ == "__main__":
    main()