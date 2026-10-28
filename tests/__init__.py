"""MoDyn test suite.

Layout
------
- ``unit/``        — fast isolated tests of single modules.
- ``integration/`` — multi-module tests (assembly, kinematics, dynamics).
- ``regression/``  — golden-data tests; references stored under ``refs/``
                     and regenerated only via ``_regen_refs.py``.

Run all tests:

    pytest

Run only unit tests:

    pytest -m unit
"""
