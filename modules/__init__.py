"""Strategy modules package.

The orchestrator is responsible for discovering strategy implementations at
runtime, therefore this package only needs to exist so Python treats
``modules/`` as a namespace.  Keeping the ``__all__`` list empty prevents us
from having to update it whenever a new strategy file is added.

Modules that rely on multiple timeframes can opt in without modifying the
orchestrator: either declare their additional requirements in
``multi_timeframe_config.MULTI_TIMEFRAME_CONFIG`` or override
``module_base.ModuleBase.process_with_timeframes`` to consume the
``extra_candles`` dictionary.
"""

__all__: list[str] = []
