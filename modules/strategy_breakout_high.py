from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from .legacy_adapter import LegacySnapshotStrategy

INTERVAL_STRING = os.environ.get("INTERVAL_STRING", "1m")
LOOKBACK_BARS = int(os.environ.get("LOOKBACK_BARS", "30"))
PRICE_TOLERANCE_RATIO = float(os.environ.get("PRICE_TOLERANCE_RATIO", "0.0008"))
MIN_PRICE = float(os.environ.get("MIN_PRICE", "0.3"))
MAX_PRICE = float(os.environ.get("MAX_PRICE", "5000"))
DEFAULT_QTY = float(os.environ.get("DEFAULT_QTY", "1.0"))
TRAILING_TYPE = os.environ.get("TRAILING_TYPE", "percent")
TRAILING_VALUE = float(os.environ.get("TRAILING_VALUE", "0.01"))
COOLDOWN_SEC = int(os.environ.get("COOLDOWN_SEC", "300"))
VOL_MULTIPLIER = float(os.environ.get("VOL_MULTIPLIER", "1.5"))
BODY_MULTIPLIER = float(os.environ.get("BODY_MULTIPLIER", "1.5"))
RANGE_MULTIPLIER = float(os.environ.get("RANGE_MULTIPLIER", "1.2"))
CLOSE_TO_HIGH_MAX_PCT = float(os.environ.get("CLOSE_TO_HIGH_MAX_PCT", "0.15"))
SMA_SHORT = int(os.environ.get("SMA_SHORT", "20"))
SMA_LONG = int(os.environ.get("SMA_LONG", "50"))
ATR_LEN = int(os.environ.get("ATR_LEN", "14"))
MIN_BREAK_ATR_MULT = float(os.environ.get("MIN_BREAK_ATR_MULT", "0.15"))
MAX_STOP_PCT = float(os.environ.get("MAX_STOP_PCT", "0.012"))
MIN_RET_PCT = float(os.environ.get("MIN_RET_PCT", "0.001"))


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

        window = arr[-(LOOKBACK_BARS + 1) : -1]
        last = arr[-1]
        prev = arr[-2]

        last_open = float(last["open"])
        last_close = float(last["close"])
        last_high = float(last["high"])
        last_low = float(last["low"])
        last_vol = float(last["volume"])

        if not (MIN_PRICE <= last_close <= MAX_PRICE):
            continue

        highs = [x["high"] for x in window]
        reference_high = max(highs) if highs else None
        if reference_high is None or reference_high <= 0.0:
            continue

        ranges = [float(x["high"]) - float(x["low"]) for x in window if float(x["high"]) >= float(x["low"])]
        med_range = median(ranges)
        if med_range is None or med_range <= 0.0:
            continue

        vols = [float(x["volume"]) for x in window]
        m_vol = median(vols) or 0.0
        if m_vol <= 0.0:
            continue

        bodies = [abs(float(x["close"]) - float(x["open"])) for x in window]
        m_body = median(bodies) or 0.0
        b_last = abs(last_close - last_open)
        last_range = last_high - last_low
        ret_pct = (last_close / float(prev["close"])) - 1.0

        s_short = sma([x["close"] for x in arr], SMA_SHORT)
        s_long = sma([x["close"] for x in arr], SMA_LONG)
        atr_val = atr_from_bars(arr, ATR_LEN)
        if s_short is None or s_long is None or atr_val is None:
            continue

        if last_close < reference_high * (1.0 + PRICE_TOLERANCE_RATIO):
            continue
        if not (last_vol >= m_vol * VOL_MULTIPLIER):
            continue
        if not (b_last >= m_body * BODY_MULTIPLIER):
            continue
        if not (last_range >= med_range * RANGE_MULTIPLIER):
            continue
        if not (ret_pct >= MIN_RET_PCT):
            continue
        if not (last_close >= s_short >= s_long):
            continue

        dist_to_high = (last_high - last_close) / last_range if last_range > 0 else 1.0
        if dist_to_high > CLOSE_TO_HIGH_MAX_PCT:
            continue

        break_depth = last_close - reference_high
        if not (break_depth >= MIN_BREAK_ATR_MULT * atr_val):
            continue

        stop_loss = last_low
        stop_pct = abs(stop_loss / last_close - 1.0)
        if stop_pct > MAX_STOP_PCT:
            continue

        key = f"breakout_high:{INTERVAL_STRING}:{symbol}:long"
        last_ts = int(mod_state.get(key, 0))
        if now_sec - last_ts < COOLDOWN_SEC:
            continue

        signals.append(
            {
                "symbol": symbol,
                "direction": "long",
                "action": "enter",
                "price": last_close,
                "quantity": DEFAULT_QTY,
                "stop_loss": float(stop_loss),
                "trailing": {"type": TRAILING_TYPE, "value": TRAILING_VALUE},
                "time": now_ms,
                "meta": {
                    "pattern": "breakout_high",
                    "reference_high": float(reference_high),
                    "lookback_bars": LOOKBACK_BARS,
                    "tolerance": PRICE_TOLERANCE_RATIO,
                    "last_volume": float(last_vol),
                    "median_volume": float(m_vol),
                    "body_last": float(b_last),
                    "median_body": float(m_body),
                    "last_range": float(last_range),
                    "median_range": float(med_range),
                    "ret_pct": float(ret_pct),
                    "sma_short": float(s_short),
                    "sma_long": float(s_long),
                    "atr": float(atr_val),
                    "break_depth": float(break_depth),
                    "stop_pct": float(stop_pct),
                },
            }
        )
        mod_state[key] = now_sec

    return signals


class BreakoutHighStrategy(LegacySnapshotStrategy):
    """Momentum breakout strategy looking for fresh highs."""

    def __init__(self, client) -> None:
        lookback = max(LOOKBACK_BARS + 2, SMA_LONG + 3, ATR_LEN + 3)
        super().__init__(
            client,
            name="Breakout High",
            abbreviation="BRKH",
            interval=INTERVAL_STRING,
            lookback=lookback,
            detect_func=detect_signals,
        )


__all__ = ["BreakoutHighStrategy", "detect_signals"]
