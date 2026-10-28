"""Unit tests for ``modyn.forces``.

Test catalogue
--------------
test_gravity_zero_force
    Gravity(0,0,0) → Q_e = 0.
test_gravity_downward
    Gravity((0,0,-g)) on one body → Q_R = m*g downward, Q_p = 0.
test_gravity_sum_two_bodies
    Two bodies; gravity assembles correct blocks for each.
test_point_force_at_com
    PointForce at marker located at body COM (s'=0) → Q_R=F, Q_p=0.
test_point_force_off_com
    PointForce off-COM; Q_p = 2 G^T (s' × A^T F) matches finite-difference.
test_point_force_callable
    Callable force is evaluated correctly at t=0.5.
test_user_force
    UserForce wraps an arbitrary (layout,t) callback.
test_to_force_fn_sums
    to_force_fn sums Gravity + PointForce correctly.
test_to_force_fn_zero_forces
    to_force_fn with empty list returns zeros.
"""

from __future__ import annotations

import numpy as np
import pytest

from modyn.dynamics.coordinates import SystemCoordinates
from modyn.forces import Gravity, PointForce, UserForce, to_force_fn
from modyn.math import quaternions as _q
from modyn.model.body import RigidBody
from modyn.model.marker import Marker

pytestmark = pytest.mark.unit


# ─── helpers ──────────────────────────────────────────────────────────────────


def _body(*, mass=2.0, p=(1, 0, 0, 0), R=(0, 0, 0)):
    return RigidBody(
        mass=mass,
        inertia=np.eye(3),
        position=R,
        quaternion=p,
        velocity=(0, 0, 0),
    )


# ─── Gravity ──────────────────────────────────────────────────────────────────


def test_gravity_zero_force():
    body = _body()
    layout = SystemCoordinates([body])
    g = Gravity(g_vec=(0.0, 0.0, 0.0))
    Qe = g.generalized_force(layout, t=0.0)
    np.testing.assert_array_equal(Qe, 0.0)


def test_gravity_downward():
    mass = 3.0
    g_val = 9.81
    body = _body(mass=mass)
    layout = SystemCoordinates([body])
    g = Gravity(g_vec=(0.0, 0.0, -g_val))
    Qe = g.generalized_force(layout, t=0.0)

    expected_R = np.array([0.0, 0.0, -mass * g_val])
    np.testing.assert_allclose(Qe[:3], expected_R)
    np.testing.assert_array_equal(Qe[3:7], 0.0)


def test_gravity_sum_two_bodies():
    m1, m2 = 1.0, 4.0
    g_vec = np.array([0.0, 0.0, -9.81])
    b1 = _body(mass=m1)
    b2 = _body(mass=m2, R=(1.0, 0.0, 0.0))
    layout = SystemCoordinates([b1, b2])
    g = Gravity(g_vec=g_vec)
    Qe = g.generalized_force(layout, t=0.0)

    np.testing.assert_allclose(Qe[0:3], m1 * g_vec)
    np.testing.assert_array_equal(Qe[3:7], 0.0)
    np.testing.assert_allclose(Qe[7:10], m2 * g_vec)
    np.testing.assert_array_equal(Qe[10:14], 0.0)


# ─── PointForce ───────────────────────────────────────────────────────────────


def test_point_force_at_com():
    """Force at COM (s'=0) → Q_R = F, Q_p = 0."""
    body = _body()
    layout = SystemCoordinates([body])
    marker = Marker(body, position=(0.0, 0.0, 0.0))
    F = np.array([10.0, 0.0, 0.0])
    pf = PointForce(marker, F)
    Qe = pf.generalized_force(layout, t=0.0)

    np.testing.assert_allclose(Qe[:3], F)
    np.testing.assert_allclose(Qe[3:7], 0.0, atol=1e-14)


def test_point_force_off_com():
    """Q_p = 2 G^T (s' × A^T F); validate with direct formula."""
    p_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # identity rotation
    body = _body(p=p_quat)
    layout = SystemCoordinates([body])
    s_body = np.array([1.0, 0.0, 0.0])
    marker = Marker(body, position=s_body)
    F = np.array([0.0, 0.0, 5.0])
    pf = PointForce(marker, F)
    Qe = pf.generalized_force(layout, t=0.0)

    # At identity rotation A=I, A^T F = F
    tau_body = np.cross(s_body, F)  # (0,0,0)×(0,0,5) → (0*5-0*0, 0*0-1*5, 1*0-0*0) = (0,-5,0)
    G = _q.G(p_quat)
    expected_Qp = 2.0 * G.T @ tau_body

    np.testing.assert_allclose(Qe[:3], F)
    np.testing.assert_allclose(Qe[3:7], expected_Qp, atol=1e-14)


def test_point_force_callable():
    body = _body()
    layout = SystemCoordinates([body])
    marker = Marker(body, position=(0.0, 0.0, 0.0))
    F_t = lambda t: np.array([t, 0.0, 0.0])  # noqa: E731
    pf = PointForce(marker, F_t)
    Qe = pf.generalized_force(layout, t=2.0)
    np.testing.assert_allclose(Qe[:3], [2.0, 0.0, 0.0])


# ─── UserForce ────────────────────────────────────────────────────────────────


def test_user_force():
    body = _body()
    layout = SystemCoordinates([body])

    def my_fn(lay, t):
        return np.ones(lay.n_coords) * t

    uf = UserForce(my_fn)
    Qe = uf.generalized_force(layout, t=3.0)
    np.testing.assert_array_equal(Qe, 3.0)


# ─── to_force_fn ──────────────────────────────────────────────────────────────


def test_to_force_fn_sums():
    mass = 2.0
    g_val = 9.81
    body = _body(mass=mass)
    layout = SystemCoordinates([body])
    marker = Marker(body, position=(0.0, 0.0, 0.0))
    F_ext = np.array([5.0, 0.0, 0.0])

    forces = [Gravity(g_vec=(0.0, 0.0, -g_val)), PointForce(marker, F_ext)]
    fn = to_force_fn(forces, layout)

    q = layout.assemble_q()
    qdot = layout.assemble_qdot()
    Qe = fn(0.0, q, qdot)

    # Expected: gravity (R-block) + point force (R-block)
    np.testing.assert_allclose(Qe[:3], F_ext + np.array([0.0, 0.0, -mass * g_val]))
    np.testing.assert_array_equal(Qe[3:7], 0.0)


def test_to_force_fn_zero_forces():
    body = _body()
    layout = SystemCoordinates([body])
    fn = to_force_fn([], layout)
    q = layout.assemble_q()
    qdot = layout.assemble_qdot()
    Qe = fn(0.0, q, qdot)
    np.testing.assert_array_equal(Qe, 0.0)
