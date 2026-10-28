"""Tests for ``modyn.math.quaternions``.

These tests cover the algebraic and kinematic identities that the rest of
MoDyn will rely on. A bug here propagates to every dynamic simulation,
so the suite is intentionally redundant.
"""

from __future__ import annotations

import numpy as np
import pytest

from modyn.math.quaternions import (
    A,
    E,
    G,
    as_quat,
    conjugate,
    from_axis_angle,
    identity,
    normalize,
    omega_body_to_pdot,
    omega_to_pdot,
    pdot_to_omega,
    pdot_to_omega_body,
    qmul,
    rotate,
    to_axis_angle,
)
from modyn.math.vectors import skew

pytestmark = pytest.mark.unit


# ─── Helpers ──────────────────────────────────────────────────────────


def _random_unit_quats(rng: np.random.Generator, n: int = 30) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _ in range(n):
        v = rng.standard_normal(4)
        out.append(v / np.linalg.norm(v))
    return out


# ─── Construction / coercion ──────────────────────────────────────────


def test_identity_is_unit_and_zero_angle():
    p = identity()
    assert np.linalg.norm(p) == pytest.approx(1.0)
    np.testing.assert_allclose(A(p), np.eye(3))


def test_as_quat_invalid_length():
    with pytest.raises(ValueError):
        as_quat([1, 0, 0])


def test_normalize_rejects_zero():
    with pytest.raises(ValueError):
        normalize(np.zeros(4))


# ─── Algebraic identities ─────────────────────────────────────────────


def test_qmul_identity_is_neutral():
    rng = np.random.default_rng(0)
    for p in _random_unit_quats(rng):
        np.testing.assert_allclose(qmul(p, identity()), p, atol=1e-14)
        np.testing.assert_allclose(qmul(identity(), p), p, atol=1e-14)


def test_conjugate_is_inverse_for_unit_quaternions():
    rng = np.random.default_rng(1)
    for p in _random_unit_quats(rng):
        prod = qmul(p, conjugate(p))
        np.testing.assert_allclose(prod, identity(), atol=1e-14)


def test_qmul_composes_rotations():
    """``A(p ⊗ q) = A(p) @ A(q)``."""
    rng = np.random.default_rng(2)
    qs = _random_unit_quats(rng, 15)
    for p in qs:
        for q in qs:
            np.testing.assert_allclose(A(qmul(p, q)), A(p) @ A(q), atol=1e-13)


# ─── Rotation matrix properties ───────────────────────────────────────


def test_A_is_orthogonal_with_det_one():
    rng = np.random.default_rng(3)
    for p in _random_unit_quats(rng):
        R = A(p)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-13)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-13)


def test_rotate_matches_matrix_multiplication():
    rng = np.random.default_rng(4)
    for p in _random_unit_quats(rng, 15):
        v = rng.standard_normal(3)
        np.testing.assert_allclose(rotate(p, v), A(p) @ v, atol=1e-13)


def test_rotate_is_isometric():
    rng = np.random.default_rng(5)
    for p in _random_unit_quats(rng, 15):
        v = rng.standard_normal(3)
        assert np.linalg.norm(rotate(p, v)) == pytest.approx(np.linalg.norm(v), abs=1e-13)


# ─── Axis-angle conversions ───────────────────────────────────────────


@pytest.mark.parametrize("axis_idx", [0, 1, 2])
def test_axis_angle_around_canonical_axes(axis_idx):
    axis = np.eye(3)[axis_idx]
    theta = 0.7
    p = from_axis_angle(axis, theta)
    # Rotating the axis itself yields the axis back.
    np.testing.assert_allclose(rotate(p, axis), axis, atol=1e-14)
    # The other two axes get rotated by ``theta`` in the orthogonal plane.
    other = np.eye(3)[(axis_idx + 1) % 3]
    expected = np.cos(theta) * other + np.sin(theta) * np.cross(axis, other)
    np.testing.assert_allclose(rotate(p, other), expected, atol=1e-14)


def test_to_from_axis_angle_round_trip():
    rng = np.random.default_rng(6)
    for _ in range(20):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)
        p = from_axis_angle(axis, angle)
        axis_r, angle_r = to_axis_angle(p)
        # Sign convention: (axis, θ) and (-axis, -θ) describe the same rotation.
        if np.dot(axis_r, axis) < 0:
            axis_r = -axis_r
            angle_r = -angle_r
        np.testing.assert_allclose(axis_r, axis, atol=1e-13)
        assert angle_r == pytest.approx(angle, abs=1e-13)


def test_to_axis_angle_identity():
    axis, angle = to_axis_angle(identity())
    np.testing.assert_allclose(axis, [1.0, 0.0, 0.0])
    assert angle == pytest.approx(0.0)


# ─── E and G matrix properties (Shabana §2.8) ─────────────────────────


def test_E_and_G_orthonormality():
    """``E E^T = G G^T = I`` for unit ``p``."""
    rng = np.random.default_rng(7)
    for p in _random_unit_quats(rng):
        np.testing.assert_allclose(E(p) @ E(p).T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(G(p) @ G(p).T, np.eye(3), atol=1e-14)


def test_E_and_G_kill_p():
    """``E p = 0`` and ``G p = 0`` for unit ``p``."""
    rng = np.random.default_rng(8)
    for p in _random_unit_quats(rng):
        np.testing.assert_allclose(E(p) @ p, np.zeros(3), atol=1e-14)
        np.testing.assert_allclose(G(p) @ p, np.zeros(3), atol=1e-14)


def test_body_global_omega_relation():
    """``ω = A(p) ω'`` (body angular velocity into global frame)."""
    rng = np.random.default_rng(9)
    for p in _random_unit_quats(rng):
        # Build a consistent ṗ from a random global ω.
        omega = rng.standard_normal(3)
        pdot = omega_to_pdot(p, omega)
        omega_body = pdot_to_omega_body(p, pdot)
        np.testing.assert_allclose(A(p) @ omega_body, omega, atol=1e-12)


def test_omega_to_pdot_round_trip_global():
    rng = np.random.default_rng(10)
    for p in _random_unit_quats(rng):
        omega = rng.standard_normal(3)
        pdot = omega_to_pdot(p, omega)
        # ṗ must be tangent to the unit sphere.
        assert abs(p @ pdot) < 1e-13
        # And recovers ω through 2 E ṗ.
        np.testing.assert_allclose(pdot_to_omega(p, pdot), omega, atol=1e-13)


def test_omega_to_pdot_round_trip_body():
    rng = np.random.default_rng(11)
    for p in _random_unit_quats(rng):
        omega_body = rng.standard_normal(3)
        pdot = omega_body_to_pdot(p, omega_body)
        assert abs(p @ pdot) < 1e-13
        np.testing.assert_allclose(pdot_to_omega_body(p, pdot), omega_body, atol=1e-13)


def test_unit_norm_preserved_at_first_order():
    """If ``ṗ`` comes from any ``ω``, then ``d/dt ||p||² = 2 p^T ṗ ≈ 0``."""
    rng = np.random.default_rng(12)
    for p in _random_unit_quats(rng):
        omega = rng.standard_normal(3)
        pdot = omega_to_pdot(p, omega)
        assert abs(2.0 * p @ pdot) < 1e-13


# ─── A(p) related identity (Shabana eq. 2.111) ────────────────────────


def test_pdot_compatible_with_Adot():
    """``Ȧ(p) = 2 ω̃ A(p)`` when ``ω = 2 E ṗ`` — verified by finite differences."""
    rng = np.random.default_rng(13)
    h = 1e-7
    for p in _random_unit_quats(rng, 10):
        omega = rng.standard_normal(3)
        pdot = omega_to_pdot(p, omega)

        # Move along ṗ and renormalize (stay on the sphere).
        p_plus = p + h * pdot
        p_plus /= np.linalg.norm(p_plus)
        p_minus = p - h * pdot
        p_minus /= np.linalg.norm(p_minus)

        Adot_num = (A(p_plus) - A(p_minus)) / (2.0 * h)
        Adot_ana = skew(omega) @ A(p)
        np.testing.assert_allclose(Adot_num, Adot_ana, atol=1e-6)
