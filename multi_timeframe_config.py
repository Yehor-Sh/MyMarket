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

By default the configuration is empty so that the caching layer is agnostic to
which modules are registered.  Individual strategies should either declare an
entry here or provide the extra timeframe information explicitly when
instantiating :class:`module_base.ModuleBase`.
"""

from __future__ import annotations

from typing import Dict

MultiTimeframeConfig = Dict[str, Dict[str, int]]

# NOTE: Projects can tailor this mapping to describe the additional timeframe
# requirements for each strategy module.  The default is intentionally empty so
# that the tester does not assume the presence of specific built-in modules.
MULTI_TIMEFRAME_CONFIG: MultiTimeframeConfig = {}

__all__ = ["MULTI_TIMEFRAME_CONFIG", "MultiTimeframeConfig"]
