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

# NOTE: The mapping is intentionally empty by default.  Individual projects can
# populate it with the requirements of their own strategy modules.
MULTI_TIMEFRAME_CONFIG: MultiTimeframeConfig = {}

__all__ = ["MULTI_TIMEFRAME_CONFIG", "MultiTimeframeConfig"]
