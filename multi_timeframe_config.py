"""Configuration describing multi-timeframe requirements for modules.

The :data:`MULTI_TIMEFRAME_CONFIG` mapping associates a module abbreviation
with the additional timeframe data it needs in order to confirm a signal.

Schema
======
``MULTI_TIMEFRAME_CONFIG`` is structured as ``{abbreviation: {interval: lookback}}``
where:

* ``abbreviation`` is the short code used by :class:`module_base.ModuleBase`
  to identify the module (for example ``"RSI"``).
* ``interval`` is the Binance-compatible interval string (for example
  ``"4h"`` or ``"1d"``).
* ``lookback`` is an integer describing how many candles should be requested
  for that interval.

Modules that require additional confirmation candles should either declare
an entry here or provide the extra timeframe information explicitly when
instantiating :class:`module_base.ModuleBase`.
"""

from __future__ import annotations

from typing import Dict

MultiTimeframeConfig = Dict[str, Dict[str, int]]

# NOTE: Projects can tailor this mapping to describe the additional timeframe
# requirements for each strategy module.  The defaults below align the bundled
# strategies with the multi-timeframe rules described in the documentation.
MULTI_TIMEFRAME_CONFIG: MultiTimeframeConfig = {
    # RSI Divergence (DIV) – confirm divergence and RSI extremes on H1.
    "DIV": {"1h": 220},
    # Pin Bar + Level + EMA (PIN) – key levels are derived from H1.
    "PIN": {"1h": 200},
    # Engulfing + RSI (ENG) – validate trend and momentum on M30/H1.
    "ENG": {"30m": 160, "1h": 160},
    # ATR + EMA Breakout (BRK) – trend alignment on H1 via EMA20/EMA50.
    "BRK": {"1h": 200},
    # Inside Bar Breakout (INS) – confirm EMA trend on M30.
    "INS": {"30m": 160},
}

__all__ = ["MULTI_TIMEFRAME_CONFIG", "MultiTimeframeConfig"]
