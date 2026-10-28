"""Tests for ``modyn.math.rotations``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.math import quaternions as q
from modyn.math.rotations import (
    from_euler,
    is_rotation_matrix,
    matrix_to_quat,
    quat_to_matrix,
    rotx,
    roty,
    rotz,
    to_euler,
)

pytestmark = pytest.mark.unit


CARDAN = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
PROPER = ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]
ALL_SEQ = CARDAN + PROPER


@pytest.mark.parametrize("rot,axis", [(rotx, 0), (roty, 1), (rotz, 2)])
def test_elementary_rotations_are_rotation_matrices(rot, axis):
    for theta in np.linspace(-np.pi, np.pi, 9):
        R = rot(theta)
        assert is_rotation_matrix(R)
        # Axis is invariant.
        e = np.eye(3)[axis]
        np.testing.assert_allclose(R @ e, e, atol=1e-15)


@pytest.mark.parametrize("seq", ALL_SEQ)
def test_from_euler_produces_rotation_matrices(seq):
    rng = np.random.default_rng(hash(seq) & 0xFFFF)
    for _ in range(5):
        a = rng.uniform(-1.0, 1.0, size=3)
        R = from_euler(seq, a)
        assert is_rotation_matrix(R)


@pytest.mark.parametrize("seq", ALL_SEQ)
def test_to_from_euler_round_trip_away_from_singularity(seq):
    """Using middle angles in a safe range, ``from_euler ∘ to_euler`` is identity."""
    rng = np.random.default_rng(hash(seq) & 0xFFFF)
    proper = seq[0] == seq[2]
    # Stay away from singular middle-angle values:
    # proper Euler is singular at middle = 0 or ±π; Cardan at middle = ±π/2.
    a1_range = (0.3, np.pi - 0.3) if proper else (-np.pi / 2 + 0.3, np.pi / 2 - 0.3)

    for _ in range(10):
        a = np.array(
            [
                rng.uniform(-np.pi + 0.01, np.pi - 0.01),
                rng.uniform(*a1_range),
                rng.uniform(-np.pi + 0.01, np.pi - 0.01),
            ]
        )
        R = from_euler(seq, a)
        a_rec = to_euler(seq, R)
        R_rec = from_euler(seq, a_rec)
        np.testing.assert_allclose(R_rec, R, atol=1e-12)


def test_invalid_sequence_rejected():
    with pytest.raises(ValueError):
        from_euler("abc", [0, 0, 0])
    with pytest.raises(ValueError):
        from_euler("xx z", [0, 0, 0])
    with pytest.raises(ValueError):
        from_euler("xxz", [0, 0, 0])


def test_matrix_to_quat_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(30):
        v = rng.standard_normal(4)
        p = v / np.linalg.norm(v)
        if p[0] < 0:
            p = -p
        R = q.A(p)
        p_rec = matrix_to_quat(R)
        np.testing.assert_allclose(p_rec, p, atol=1e-13)


def test_matrix_to_quat_handles_identity():
    p = matrix_to_quat(np.eye(3))
    np.testing.assert_allclose(p, q.identity(), atol=1e-15)


def test_matrix_to_quat_handles_180_deg_rotations():
    # 180° rotations are the failure case of naive trace-based formulas.
    for axis in np.eye(3):
        R = -np.eye(3) + 2.0 * np.outer(axis, axis)
        p = matrix_to_quat(R)
        np.testing.assert_allclose(quat_to_matrix(p), R, atol=1e-13)


def test_is_rotation_matrix_rejects_reflections():
    R = np.diag([1.0, 1.0, -1.0])  # reflection, det = -1
    assert not is_rotation_matrix(R)


def test_is_rotation_matrix_rejects_non_orthogonal():
    R = np.eye(3) + 0.1
    assert not is_rotation_matrix(R)


def test_quat_to_matrix_alias():
    p = q.from_axis_angle([0, 0, 1], 0.7)
    np.testing.assert_array_equal(quat_to_matrix(p), q.A(p))
