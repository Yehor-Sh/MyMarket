from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from .legacy_adapter import LegacySnapshotStrategy

INTERVAL_STRING = os.environ.get("INTERVAL_STRING", "1m")
LOOKBACK_BARS = int(os.environ.get("LOOKBACK_BARS", "50"))
VOLUME_MULTIPLIER = float(os.environ.get("VOLUME_MULTIPLIER", "3.0"))
MIN_PRICE = float(os.environ.get("MIN_PRICE", "0.3"))
MAX_PRICE = float(os.environ.get("MAX_PRICE", "5000"))
BODY_MULTIPLIER = float(os.environ.get("BODY_MULTIPLIER", "2.0"))
RANGE_MULTIPLIER = float(os.environ.get("RANGE_MULTIPLIER", "1.5"))
MIN_RANGE_PCT = float(os.environ.get("MIN_RANGE_PCT", "0.001"))
MIN_RET_PCT = float(os.environ.get("MIN_RET_PCT", "0.002"))
SMA_LEN = int(os.environ.get("SMA_LEN", "20"))
COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "300"))
DEFAULT_QTY = float(os.environ.get("DEFAULT_QTY", "1.0"))
TRAILING_TYPE = os.environ.get("TRAILING_TYPE", "percent")
TRAILING_VALUE = float(os.environ.get("TRAILING_VALUE", "0.015"))


def to_closed_numeric(bars: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for b in bars:
        if not b.get("is_closed"):
            continue
        try:
            out.append(
                {
                    "open_time": float(b["open_time"]),
                    "open": float(b["open"]),
                    "high": float(b["high"]),
                    "low": float(b["low"]),
                    "close": float(b["close"]),
                    "volume": float(b["volume"]),
                }
            )
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


def body_size(o: float, c: float) -> float:
    return abs(c - o)


def range_size(h: float, l: float) -> float:
    return abs(h - l)


def detect_signals(snapshot: Dict[str, Any], mod_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    symbols_map: Dict[str, List[Dict[str, Any]]] = snapshot.get("symbols", {})
    now_ms = int(time.time() * 1000)
    now_sec = int(time.time())

    for symbol, bars in symbols_map.items():
        arr = to_closed_numeric(bars)
        if len(arr) < max(LOOKBACK_BARS + 1, SMA_LEN + 1):
            continue

        recent = arr[-(LOOKBACK_BARS + 1) :]
        prices = [x["close"] for x in recent]
        vols = [x["volume"] for x in recent[:-1]]
        last = recent[-1]
        prev = recent[-2]
        last_close = float(last["close"])
        last_open = float(last["open"])
        last_high = float(last["high"])
        last_low = float(last["low"])
        last_vol = float(last["volume"])
        prev_close = float(prev["close"])

        if not (MIN_PRICE <= last_close <= MAX_PRICE):
            continue

        m_vol = median(vols)
        if m_vol is None or m_vol <= 0.0:
            continue

        b_last = body_size(last_open, last_close)
        bodies = [body_size(x["open"], x["close"]) for x in recent[:-1]]
        m_body = median(bodies) or 0.0

        r_last = range_size(last_high, last_low)
        ranges = [range_size(x["high"], x["low"]) for x in recent[:-1]]
        m_range = median(ranges) or 0.0

        ret_pct = (last_close / prev_close) - 1.0
        rng_pct = r_last / last_close if last_close > 0 else 0.0

        sma_val = sma([x["close"] for x in arr], SMA_LEN)
        if sma_val is None:
            continue

        if not (last_vol >= m_vol * VOLUME_MULTIPLIER):
            continue
        if not (b_last >= m_body * BODY_MULTIPLIER):
            continue
        if not (r_last >= m_range * RANGE_MULTIPLIER or rng_pct >= MIN_RANGE_PCT):
            continue
        if not (abs(ret_pct) >= MIN_RET_PCT):
            continue

        direction = "long" if last_close >= median(prices) else "short"
        if direction == "long" and not (last_close >= sma_val):
            continue
        if direction == "short" and not (last_close <= sma_val):
            continue

        ms_key = f"volume_surge:{INTERVAL_STRING}:{symbol}:{direction}"
        last_ts = int(mod_state.get(ms_key, 0))
        if now_sec - last_ts < COOLDOWN_SEC:
            continue

        signals.append(
            {
                "symbol": symbol,
                "direction": direction,
                "action": "enter",
                "price": last_close,
                "quantity": DEFAULT_QTY,
                "stop_loss": None,
                "trailing": {"type": TRAILING_TYPE, "value": TRAILING_VALUE},
                "time": now_ms,
                "meta": {
                    "pattern": "volume_surge",
                    "lookback": LOOKBACK_BARS,
                    "multiplier": VOLUME_MULTIPLIER,
                    "last_volume": last_vol,
                    "median_volume": m_vol,
                    "body_last": b_last,
                    "median_body": m_body,
                    "range_last": r_last,
                    "median_range": m_range,
                    "ret_pct": ret_pct,
                    "sma_len": SMA_LEN,
                    "sma": sma_val,
                },
            }
        )
        mod_state[ms_key] = now_sec

    return signals


class VolumeSurgeStrategy(LegacySnapshotStrategy):
    """Detect strong volume spikes accompanied by directional moves."""

    def __init__(self, client) -> None:
        lookback = max(LOOKBACK_BARS + 2, SMA_LEN + 2)
        super().__init__(
            client,
            name="Volume Surge",
            abbreviation="VSRG",
            interval=INTERVAL_STRING,
            lookback=lookback,
            detect_func=detect_signals,
        )


__all__ = ["VolumeSurgeStrategy", "detect_signals"]
