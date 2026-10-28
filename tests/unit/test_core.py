"""Tests for ``modyn.core``."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from modyn.core import (
    GRAVITY_SI,
    SI,
    Base,
    UnitSystem,
    configure_logging,
    ensure_finite,
    ensure_non_negative,
    ensure_positive,
    ensure_shape,
    ensure_symmetric_positive_definite,
    get_logger,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- Base


class _A(Base):
    pass


class _B(Base):
    pass


@pytest.fixture(autouse=True)
def _reset_counters():
    Base.reset_all_counts()
    yield
    Base.reset_all_counts()


def test_base_id_increments_per_subclass():
    a1, a2, a3 = _A(), _A(), _A()
    b1, b2 = _B(), _B()
    assert (a1.id, a2.id, a3.id) == (1, 2, 3)
    assert (b1.id, b2.id) == (1, 2)
    assert _A.count() == 3
    assert _B.count() == 2


def test_base_default_name_uses_class_name():
    a = _A()
    assert a.name == "_A#1"


def test_base_custom_name_used():
    b = _B(name="left_wheel")
    assert b.name == "left_wheel"
    b.name = "right_wheel"
    assert b.name == "right_wheel"


def test_reset_count_is_per_class():
    _A(), _A()
    _B()
    _A.reset_count()
    assert _A.count() == 0
    assert _B.count() == 1


# --------------------------------------------------------------- utils


def test_as_float_array_validators():
    a = ensure_shape([1, 2, 3], (3,))
    assert a.shape == (3,) and a.dtype == np.float64
    with pytest.raises(ValueError):
        ensure_shape([1, 2], (3,))


def test_ensure_finite_rejects_nan_inf():
    ensure_finite([1.0, 2.0])
    with pytest.raises(ValueError):
        ensure_finite([1.0, np.nan])
    with pytest.raises(ValueError):
        ensure_finite([np.inf, 0.0])


def test_ensure_positive_and_non_negative():
    assert ensure_positive(3.5) == 3.5
    with pytest.raises(ValueError):
        ensure_positive(0.0)
    with pytest.raises(ValueError):
        ensure_positive(-1.0)
    assert ensure_non_negative(0.0) == 0.0
    with pytest.raises(ValueError):
        ensure_non_negative(-1e-12)


def test_ensure_spd_accepts_identity_and_symmetrizes():
    M = np.eye(3) + 1e-15 * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    A = ensure_symmetric_positive_definite(M)
    np.testing.assert_allclose(A, A.T, atol=0)


def test_ensure_spd_rejects_non_square_and_non_pd():
    with pytest.raises(ValueError):
        ensure_symmetric_positive_definite(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        ensure_symmetric_positive_definite(np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        ensure_symmetric_positive_definite(np.array([[1.0, 2.0], [0.0, 1.0]]))  # asymmetric


# --------------------------------------------------------------- units


def test_default_si_system():
    assert SI.length == "m"
    assert SI.mass == "kg"
    assert SI.time == "s"
    assert SI.gravity == GRAVITY_SI
    assert "kg" in SI.force and "m" in SI.force
    assert "kg" in SI.torque


def test_unit_system_is_frozen():
    with pytest.raises(AttributeError):
        SI.length = "ft"  # type: ignore[misc]


def test_custom_unit_system():
    U = UnitSystem(length="mm", mass="g", time="ms", gravity=9806.65)
    assert U.gravity == 9806.65


# ------------------------------------------------------------- logging


def test_configure_logging_idempotent():
    L1 = configure_logging(logging.WARNING)
    n1 = len(L1.handlers)
    L2 = configure_logging(logging.WARNING)
    n2 = len(L2.handlers)
    assert L1 is L2
    assert n1 == n2


def test_get_logger_namespacing():
    root = get_logger()
    child = get_logger("solver")
    assert root.name == "modyn"
    assert child.name == "modyn.solver"
