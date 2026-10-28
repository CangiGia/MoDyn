"""Auto-reset fixtures shared by all unit tests.

The ``Ground`` singleton and ``Base`` counter need to be reset between
tests for full isolation. This fixture is autouse-scoped to ``tests/unit/``
only so it does not affect integration / regression suites that may want
to keep state across tests.
"""

from __future__ import annotations

import pytest

from modyn.core.base import Base
from modyn.model.ground import Ground


@pytest.fixture(autouse=True)
def _reset_singletons():
    Ground._instance = None
    Base.reset_all_counts()
    yield
    Ground._instance = None
    Base.reset_all_counts()
