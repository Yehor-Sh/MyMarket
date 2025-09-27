#!/usr/bin/env python3
from __future__ import annotations

# -*- coding: utf-8 -*-

import json
import os
import time
import uuid
from typing import Dict, List, Any, Optional

import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8080")
SERVER_TOKEN = os.environ.get("SERVER_TOKEN", "dev-token")
BARS_FILE_PATH = os.environ.get("BARS_FILE_PATH", "./bars_1m.json")
INTERVAL_STRING = os.environ.get("INTERVAL_STRING", "1m")
MODULE_NAME = os.environ.get("MODULE_NAME", "module_F_donchian_breakout")
MODULE_VERSION = os.environ.get("MODULE_VERSION", "1.0.0")

LOOKBACK = int(os.environ.get("LOOKBACK", "50"))
TOLERANCE = float(os.environ.get("TOLERANCE", "0.0008"))
VOL_MULTIPLIER = float(os.environ.get("VOL_MULTIPLIER", "1.4"))
COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "300"))

DEFAULT_QTY = float(os.environ.get("DEFAULT_QTY", "1.0"))
TRAILING_TYPE = "percent"
TRAILING_VALUE = float(os.environ.get("TRAILING_VALUE", "0.012"))

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

def median(a: List[float]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    s = sorted(a)
    m = n // 2
    if n % 2 == 1:
        return float(s[m])
    return float((s[m - 1] + s[m]) / 2.0)

def detect(snapshot: Dict[str, Any], mod_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    now_sec = int(time.time())
    symbols = snapshot.get("symbols", {})
    for sym, bars in symbols.items():
        arr = to_closed(bars)
        if len(arr) < LOOKBACK + 1:
            continue
        window = arr[-(LOOKBACK + 1):-1]
        last = arr[-1]
        last_close = float(last["close"])
        last_low = float(last["low"])
        last_high = float(last["high"])
        if not (MIN_PRICE <= last_close <= MAX_PRICE):
            continue
        highs = [x["high"] for x in window]
        lows = [x["low"] for x in window]
        ref_high = max(highs) if highs else None
        ref_low = min(lows) if lows else None
        if ref_high is None or ref_low is None:
            continue
        vols = [x["volume"] for x in window]
        v_med = median(vols) or 0.0
        if v_med <= 0.0:
            continue
        last_vol = float(last["volume"])
        if last_close >= ref_high * (1.0 + TOLERANCE) and last_vol >= v_med * VOL_MULTIPLIER:
            key = f"{MODULE_NAME}:{INTERVAL_STRING}:{sym}:long"
            if now_sec - int(mod_state.get(key, 0)) < COOLDOWN_SEC:
                continue
            out.append({"symbol": sym, "direction": "long", "action": "enter", "price": last_close, "quantity": DEFAULT_QTY, "stop_loss": None, "trailing": {"type": TRAILING_TYPE, "value": TRAILING_VALUE}, "time": now_ms, "meta": {"pattern": "donchian_breakout_up", "lookback": LOOKBACK, "ref_high": ref_high}})
            mod_state[key] = now_sec
        elif last_close <= ref_low * (1.0 - TOLERANCE) and last_vol >= v_med * VOL_MULTIPLIER:
            key = f"{MODULE_NAME}:{INTERVAL_STRING}:{sym}:short"
            if now_sec - int(mod_state.get(key, 0)) < COOLDOWN_SEC:
                continue
            out.append({"symbol": sym, "direction": "short", "action": "enter", "price": last_close, "quantity": DEFAULT_QTY, "stop_loss": None, "trailing": {"type": TRAILING_TYPE, "value": TRAILING_VALUE}, "time": now_ms, "meta": {"pattern": "donchian_breakout_down", "lookback": LOOKBACK, "ref_low": ref_low}})
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
        print(f"[module F] state_fetch_failed: {exc}")
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
        print(f"[module F] state_persist_failed: {exc}")


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
