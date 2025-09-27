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
MODULE_NAME = os.environ.get("MODULE_NAME", "module_K_vwap_reversion")
MODULE_VERSION = os.environ.get("MODULE_VERSION", "1.0.0")

VWAP_LEN = int(os.environ.get("VWAP_LEN", "60"))
DEV_MULT = float(os.environ.get("DEV_MULT", "1.5"))
VOL_MULTIPLIER = float(os.environ.get("VOL_MULTIPLIER", "1.2"))
COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "300"))

DEFAULT_QTY = float(os.environ.get("DEFAULT_QTY", "1.0"))
TRAILING_TYPE = "percent"
TRAILING_VALUE = float(os.environ.get("TRAILING_VALUE", "0.01"))

MIN_PRICE = float(os.environ.get("MIN_PRICE", "0.3"))
MAX_PRICE = float(os.environ.get("MAX_PRICE", "5000"))
REQUEST_TIMEOUT = 10.0
MODULE_HTTP_TIMEOUT = float(os.environ.get("MODULE_HTTP_TIMEOUT", "5.0"))

def load_json(path: str, default: Any) -> Any:
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def to_closed(bars: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out = []
    for b in bars:
        if not b.get("is_closed"):
            continue
        try:
            out.append({"open_time": float(b["open_time"]), "open": float(b["open"]), "high": float(b["high"]), "low": float(b["low"]), "close": float(b["close"]), "volume": float(b["volume"])})
        except Exception:
            continue
    return out

def vwap(prices: List[float], volumes: List[float], n: int) -> Optional[float]:
    if len(prices) < n or len(volumes) < n:
        return None
    p = prices[-n:]
    v = volumes[-n:]
    tv = sum(p[i] * v[i] for i in range(n))
    vv = sum(v)
    if vv <= 0:
        return None
    return tv / vv

def std(a: List[float]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    m = sum(a) / n
    v = sum((x - m) * (x - m) for x in a) / n
    return v ** 0.5

def detect(snapshot: Dict[str, Any], mod_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    now_sec = int(time.time())
    symbols = snapshot.get("symbols", {})
    for sym, bars in symbols.items():
        arr = to_closed(bars)
        if len(arr) < max(VWAP_LEN + 2, 60):
            continue
        closes = [x["close"] for x in arr]
        vols = [x["volume"] for x in arr]
        last_close = float(closes[-1])
        if not (MIN_PRICE <= last_close <= MAX_PRICE):
            continue
        val = vwap(closes, vols, VWAP_LEN)
        if val is None:
            continue
        devs = [abs(closes[i] - val) / val for i in range(-VWAP_LEN, 0)]
        sd = std(devs)
        v_med = sorted(vols[-VWAP_LEN:])[VWAP_LEN // 2]
        if v_med <= 0.0:
            continue
        last_vol = float(vols[-1])
        if last_vol < v_med * VOL_MULTIPLIER:
            continue
        if sd <= 0.0:
            continue
        dist = (last_close - val) / val
        if dist >= DEV_MULT * sd:
            key = f"{MODULE_NAME}:{INTERVAL_STRING}:{sym}:short"
            if now_sec - int(mod_state.get(key, 0)) >= COOLDOWN_SEC:
                out.append({"symbol": sym, "direction": "short", "action": "enter", "price": last_close, "quantity": DEFAULT_QTY, "stop_loss": None, "trailing": {"type": TRAILING_TYPE, "value": TRAILING_VALUE}, "time": now_ms, "meta": {"pattern": "vwap_reversion_short", "vwap": val, "z": dist / sd}})
                mod_state[key] = now_sec
        elif dist <= -DEV_MULT * sd:
            key = f"{MODULE_NAME}:{INTERVAL_STRING}:{sym}:long"
            if now_sec - int(mod_state.get(key, 0)) >= COOLDOWN_SEC:
                out.append({"symbol": sym, "direction": "long", "action": "enter", "price": last_close, "quantity": DEFAULT_QTY, "stop_loss": None, "trailing": {"type": TRAILING_TYPE, "value": TRAILING_VALUE}, "time": now_ms, "meta": {"pattern": "vwap_reversion_long", "vwap": val, "z": dist / sd}})
                mod_state[key] = now_sec
    return out


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
        print(f"[module K] state_fetch_failed: {exc}")
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
        print(f"[module K] state_persist_failed: {exc}")


def register(client: httpx.Client) -> str:
    r = client.post(f"{SERVER_URL}/modules/register", headers={"Authorization": f"Bearer {SERVER_TOKEN}"}, json={"module_name": MODULE_NAME, "version": MODULE_VERSION, "capabilities": {"intervals": [INTERVAL_STRING]}, "expires_in_sec": 180, "request_id": str(uuid.uuid4())}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["module_id"]

def report(client: httpx.Client, module_id: str, signals: List[Dict[str, Any]]) -> None:
    r = client.post(f"{SERVER_URL}/signals/report", headers={"Authorization": f"Bearer {SERVER_TOKEN}"}, json={"module_id": module_id, "interval": INTERVAL_STRING, "signals": signals, "request_id": str(uuid.uuid4())}, timeout=REQUEST_TIMEOUT + 5)
    r.raise_for_status()

def main() -> None:
    snapshot = load_json(BARS_FILE_PATH, {"symbols": {}, "meta": {}})
    mod_state = fetch_module_state()
    signals = detect(snapshot, mod_state)
    if not signals and os.environ.get("SEND_UPDATES_IF_NO_SIGNALS", "0") != "1":
        persist_module_state(mod_state)
        return
    with httpx.Client() as client:
        mid = register(client)
        report(client, mid, signals)
    persist_module_state(mod_state)

if __name__ == "__main__":
    main()