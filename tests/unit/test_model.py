"""Tests for ``modyn.model``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.core.base import Base
from modyn.math import quaternions as q
from modyn.model import Ground, Marker, RigidBody

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_counters():
    Base.reset_all_counts()
    # Ground is a singleton: drop it so each test gets a fresh one.
    Ground._instance = None
    yield
    Ground._instance = None
    Base.reset_all_counts()


# --------------------------------------------------------------- Ground


def test_ground_is_singleton():
    g1 = Ground()
    g2 = Ground.instance()
    assert g1 is g2


def test_ground_state_is_zero_and_immutable_geometry():
    g = Ground()
    np.testing.assert_array_equal(g.position, np.zeros(3))
    np.testing.assert_array_equal(g.quaternion, q.identity())
    np.testing.assert_array_equal(g.rotation_matrix, np.eye(3))
    np.testing.assert_array_equal(g.velocity, np.zeros(3))
    np.testing.assert_array_equal(g.omega_global, np.zeros(3))
    s = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(g.point_global(s), s)
    np.testing.assert_array_equal(g.velocity_of_point(s), np.zeros(3))
    assert g.mass == float("inf")


# ----------------------------------------------------------- RigidBody


def test_rigid_body_default_state():
    b = RigidBody(mass=2.0, inertia=np.diag([1.0, 2.0, 3.0]))
    assert b.mass == 2.0
    np.testing.assert_array_equal(b.position, np.zeros(3))
    np.testing.assert_array_equal(b.quaternion, q.identity())
    np.testing.assert_array_equal(b.rotation_matrix, np.eye(3))
    np.testing.assert_array_equal(b.q, np.array([0, 0, 0, 1, 0, 0, 0]))


def test_rigid_body_rejects_invalid_mass_and_inertia():
    with pytest.raises(ValueError):
        RigidBody(mass=0.0, inertia=np.eye(3))
    with pytest.raises(ValueError):
        RigidBody(mass=-1.0, inertia=np.eye(3))
    with pytest.raises(ValueError):
        RigidBody(mass=1.0, inertia=np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        RigidBody(mass=1.0, inertia=np.eye(2))


def test_rigid_body_quaternion_is_normalized():
    b = RigidBody(mass=1.0, inertia=np.eye(3), quaternion=[2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(b.quaternion), 1.0, atol=1e-15)


def test_rigid_body_q_round_trip():
    b = RigidBody(mass=1.0, inertia=np.eye(3))
    p = q.from_axis_angle([0, 0, 1], 0.7)
    b.q = np.concatenate(([1.0, 2.0, 3.0], p))
    np.testing.assert_allclose(b.position, [1, 2, 3])
    np.testing.assert_allclose(b.quaternion, p)


def test_rigid_body_omega_body_global_round_trip():
    p = q.from_axis_angle([0, 1, 0], 0.4)
    b = RigidBody(mass=1.0, inertia=np.eye(3), quaternion=p)
    w_global = np.array([0.1, 0.2, 0.3])
    b.omega_global = w_global
    np.testing.assert_allclose(b.omega_global, w_global, atol=1e-13)
    # omega_body equals A^T omega_global.
    np.testing.assert_allclose(b.omega_body, q.A(p).T @ w_global, atol=1e-13)


def test_rigid_body_inertia_global_consistency():
    p = q.from_axis_angle([1, 1, 0], 0.6)
    J = np.diag([1.0, 2.0, 5.0])
    b = RigidBody(mass=1.0, inertia=J, quaternion=p)
    A = q.A(p)
    np.testing.assert_allclose(b.inertia_global, A @ J @ A.T, atol=1e-13)


def test_rigid_body_point_and_velocity_global():
    p = q.from_axis_angle([0, 0, 1], np.pi / 2)
    b = RigidBody(
        mass=1.0,
        inertia=np.eye(3),
        position=[10, 0, 0],
        quaternion=p,
        velocity=[0, 0, 0],
        omega_body=q.A(p).T @ np.array([0, 0, 1.0]),  # ω_global = e_z
    )
    s_body = np.array([1.0, 0.0, 0.0])
    # Rotated by 90° about z: A s' = e_y; so global point = (10, 1, 0).
    np.testing.assert_allclose(b.point_global(s_body), [10, 1, 0], atol=1e-13)
    # v_P = ω × (A s') = e_z × e_y = -e_x.
    np.testing.assert_allclose(b.velocity_of_point(s_body), [-1, 0, 0], atol=1e-13)


# ------------------------------------------------------------- Marker


def test_marker_on_ground_is_world_fixed():
    g = Ground()
    m = Marker(g, position=[1, 2, 3])
    np.testing.assert_array_equal(m.position_global, [1, 2, 3])
    np.testing.assert_array_equal(m.orientation_global, np.eye(3))


def test_marker_follows_body():
    p = q.from_axis_angle([0, 0, 1], np.pi / 2)
    b = RigidBody(mass=1.0, inertia=np.eye(3), position=[5, 0, 0], quaternion=p)
    m = Marker(b, position=[1, 0, 0])
    np.testing.assert_allclose(m.position_global, [5, 1, 0], atol=1e-13)
    np.testing.assert_allclose(m.orientation_global, q.A(p), atol=1e-13)


def test_marker_rejects_non_rotation_orientation():
    g = Ground()
    with pytest.raises(ValueError):
        Marker(g, orientation=np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError):
        Marker(g, orientation=np.eye(2))
