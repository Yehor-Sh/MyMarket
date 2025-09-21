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
MODULE_NAME = os.environ.get("MODULE_NAME", "module_C_breakdown_low")
MODULE_VERSION = os.environ.get("MODULE_VERSION", "1.1.0")

LOOKBACK_BARS = int(os.environ.get("LOOKBACK_BARS", "30"))
PRICE_TOLERANCE_RATIO = float(os.environ.get("PRICE_TOLERANCE_RATIO", "0.0008"))
MIN_PRICE = float(os.environ.get("MIN_PRICE", "0.3"))
MAX_PRICE = float(os.environ.get("MAX_PRICE", "5000"))
DEFAULT_QTY = float(os.environ.get("DEFAULT_QTY", "1.0"))
TRAILING_TYPE = os.environ.get("TRAILING_TYPE", "percent")
TRAILING_VALUE = float(os.environ.get("TRAILING_VALUE", "0.01"))
SEND_UPDATES_IF_NO_SIGNALS = os.environ.get("SEND_UPDATES_IF_NO_SIGNALS", "0") == "1"
REQUEST_TIMEOUT = 10.0
MODULE_HTTP_TIMEOUT = float(os.environ.get("MODULE_HTTP_TIMEOUT", "5.0"))

COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "300"))
VOL_MULTIPLIER = float(os.environ.get("VOL_MULTIPLIER", "1.5"))
BODY_MULTIPLIER = float(os.environ.get("BODY_MULTIPLIER", "1.5"))
RANGE_MULTIPLIER = float(os.environ.get("RANGE_MULTIPLIER", "1.2"))
SMA_SHORT = int(os.environ.get("SMA_SHORT", "20"))
SMA_LONG = int(os.environ.get("SMA_LONG", "50"))
ATR_LEN = int(os.environ.get("ATR_LEN", "14"))
MIN_BREAK_ATR_MULT = float(os.environ.get("MIN_BREAK_ATR_MULT", "0.15"))
MAX_STOP_PCT = float(os.environ.get("MAX_STOP_PCT", "0.01"))

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

def true_range(prev_close: float, high: float, low: float) -> float:
    a = high - low
    b = abs(high - prev_close)
    c = abs(low - prev_close)
    return max(a, b, c)

def atr_from_bars(arr: List[Dict[str, float]], n: int) -> Optional[float]:
    if len(arr) < n + 1:
        return None
    trs: List[float] = []
    for i in range(-n, 0):
        prev_c = arr[i - 1]["close"]
        h = arr[i]["high"]
        l = arr[i]["low"]
        trs.append(true_range(prev_c, h, l))
    return float(sum(trs) / len(trs)) if trs else None

def body_size(o: float, c: float) -> float:
    return abs(c - o)

def range_size(h: float, l: float) -> float:
    return abs(h - l)

def detect_signals(snapshot: Dict[str, Any], mod_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    symbols_map: Dict[str, List[Dict[str, Any]]] = snapshot.get("symbols", {})
    now_ms = int(time.time() * 1000)
    now_sec = int(time.time())
    need = max(LOOKBACK_BARS + 1, SMA_LONG + 2, ATR_LEN + 2)

    for symbol, bars in symbols_map.items():
        arr = to_closed_numeric(bars)
        if len(arr) < need:
            continue

        window = arr[-(LOOKBACK_BARS + 1):-1]
        last = arr[-1]
        prev = arr[-2]

        last_close = float(last["close"])
        last_open = float(last["open"])
        last_high = float(last["high"])
        last_low = float(last["low"])
        last_vol = float(last["volume"])

        if not (MIN_PRICE <= last_close <= MAX_PRICE):
            continue

        reference_low = min(x["low"] for x in window)
        if reference_low <= 0:
            continue

        vols = [x["volume"] for x in window]
        m_vol = median(vols) or 0.0
        if m_vol <= 0.0:
            continue

        b_last = body_size(last_open, last_close)
        m_body = median([body_size(x["open"], x["close"]) for x in window]) or 0.0
        r_last = range_size(last_high, last_low)
        m_range = median([range_size(x["high"], x["low"]) for x in window]) or 0.0

        s_short = sma([x["close"] for x in arr], SMA_SHORT)
        s_long = sma([x["close"] for x in arr], SMA_LONG)
        s_long_prev = sma([x["close"] for x in arr[:-1]], SMA_LONG)

        atr_val = atr_from_bars(arr, ATR_LEN)

        if last_close > reference_low * (1.0 - PRICE_TOLERANCE_RATIO):
            continue
        if not (last_vol >= m_vol * VOL_MULTIPLIER):
            continue
        if not (b_last >= m_body * BODY_MULTIPLIER):
            continue
        if not (r_last >= m_range * RANGE_MULTIPLIER):
            continue
        if s_short is None or s_long is None or s_long_prev is None or atr_val is None:
            continue
        if not (s_short <= s_long and s_long <= s_long_prev):
            continue

        break_depth = reference_low - last_close
        if not (break_depth >= MIN_BREAK_ATR_MULT * atr_val):
            continue

        stop_loss = last_high
        stop_pct = (stop_loss / last_close) - 1.0
        if stop_pct > MAX_STOP_PCT:
            continue

        ms_key = f"{MODULE_NAME}:{INTERVAL_STRING}:{symbol}:short"
        last_ts = int(mod_state.get(ms_key, 0))
        if now_sec - last_ts < COOLDOWN_SEC:
            continue

        signals.append({
            "symbol": symbol,
            "direction": "short",
            "action": "enter",
            "price": last_close,
            "quantity": DEFAULT_QTY,
            "stop_loss": stop_loss,
            "trailing": {"type": TRAILING_TYPE, "value": TRAILING_VALUE},
            "time": now_ms,
            "meta": {
                "pattern": "breakdown_low",
                "reference_low": reference_low,
                "lookback_bars": LOOKBACK_BARS,
                "tolerance": PRICE_TOLERANCE_RATIO,
                "last_volume": last_vol,
                "median_volume": m_vol,
                "body_last": b_last,
                "median_body": m_body,
                "range_last": r_last,
                "median_range": m_range,
                "sma_short": s_short,
                "sma_long": s_long,
                "atr": atr_val,
                "break_depth": break_depth,
                "stop_pct": stop_pct
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
        print(f"[module C] state_fetch_failed: {exc}")
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
        print(f"[module C] state_persist_failed: {exc}")


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
    print("[module C]", r.json())

def main() -> None:
    snapshot = load_snapshot(BARS_FILE_PATH)
    mod_state = fetch_module_state()
    signals = detect_signals(snapshot, mod_state)
    if not signals and not SEND_UPDATES_IF_NO_SIGNALS:
        print("[module C] no_signals")
        persist_module_state(mod_state)
        return
    with httpx.Client() as client:
        module_id = register_module(client)
        report_signals(client, module_id, signals)
    persist_module_state(mod_state)

if __name__ == "__main__":
    main()