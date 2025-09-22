"""Strategy modules package.

The orchestrator is responsible for discovering strategy implementations at
runtime, therefore this package only needs to exist so Python treats
``modules/`` as a namespace.  Keeping the ``__all__`` list empty prevents us
from having to update it whenever a new strategy file is added.
"""

__all__: list[str] = []
