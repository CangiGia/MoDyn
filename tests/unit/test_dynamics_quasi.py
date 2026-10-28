"""Unit tests for ``modyn.dynamics.quasi`` (quasi-velocity formulation)."""

import numpy as np

from modyn.constraints import RevoluteJoint, SphericalJoint
from modyn.dynamics import SystemCoordinates
from modyn.dynamics import eom as _eom
from modyn.dynamics import quasi as _quasi
from modyn.math import quaternions as _q
from modyn.model.body import RigidBody
from modyn.model.ground import Ground
from modyn.model.marker import Marker
from modyn.solver.kinematics import solve_velocity


def _spherical_body():
    ground = Ground()
    J = np.array([[0.30, 0.02, 0.01], [0.02, 0.50, 0.03], [0.01, 0.03, 0.70]])
    body = RigidBody(
        mass=2.0,
        inertia=J,
        position=(0.4, 0.1, -0.2),
        quaternion=_q.from_axis_angle([1.0, 2.0, 3.0], 0.7),
        velocity=(0.3, -0.2, 0.1),
        omega_body=(0.5, -0.4, 0.9),
    )
    s_body = np.array([0.2, -0.1, 0.3])
    m_g = Marker(ground, position=body.point_global(s_body))
    m_b = Marker(body, position=s_body)
    layout = SystemCoordinates([body])
    joint = SphericalJoint(m_g, m_b)
    solve_velocity(layout, [joint])
    return body, layout, joint


def test_mass_matrix_is_constant_block_diagonal():
    body, layout, _ = _spherical_body()
    M = _quasi.mass_matrix_quasi(layout)
    np.testing.assert_allclose(M[:3, :3], 2.0 * np.eye(3))
    np.testing.assert_allclose(M[3:, 3:], body.inertia)
    np.testing.assert_allclose(M[:3, 3:], 0.0)


def test_kinematic_map_matches_omega_relations():
    body, layout, _ = _spherical_body()
    v = _quasi.assemble_velocity(layout)
    qdot = _quasi.qdot_from_velocity(layout, v)
    np.testing.assert_allclose(qdot[:3], body.velocity)
    np.testing.assert_allclose(qdot[3:], _q.omega_body_to_pdot(body.quaternion, body.omega_body))
    np.testing.assert_allclose(_quasi.velocity_from_qdot(layout, qdot), v, atol=1e-14)
    # p^T pdot = 0: the unit sphere is preserved to first order
    assert abs(body.quaternion @ qdot[3:]) < 1e-14


def test_applied_force_mapping_returns_force_and_body_torque():
    body, layout, _ = _spherical_body()
    G = _q.G(body.quaternion)
    F = np.array([1.0, -2.0, 0.5])
    tau = np.array([0.3, 0.1, -0.7])
    Qe = np.concatenate((F, 2.0 * G.T @ tau))
    np.testing.assert_allclose(_quasi.applied_to_quasi(layout, Qe), np.concatenate((F, tau)))


def test_quasi_accelerations_match_pdot_formulation():
    body, layout, joint = _spherical_body()
    Qe = np.zeros(layout.n_coords)
    Qe[:3] = (0.0, 0.0, -2.0 * 9.81)
    Qe[3:] = 2.0 * _q.G(body.quaternion).T @ np.array([0.2, -0.1, 0.4])

    qddot_ref, lam_ref = _eom.solve_accelerations(layout, [joint], Qe)
    vdot, lam = _quasi.solve_accelerations_quasi(layout, [joint], Qe)

    v = _quasi.assemble_velocity(layout)
    pdot = _q.omega_body_to_pdot(body.quaternion, v[3:])
    qddot = np.empty(7)
    qddot[:3] = vdot[:3]
    qddot[3:] = 0.5 * _q.G(body.quaternion).T @ vdot[3:] + 0.5 * _q.G(pdot).T @ v[3:]

    np.testing.assert_allclose(qddot, qddot_ref, atol=1e-10)
    np.testing.assert_allclose(lam, lam_ref[: joint.n_eq], atol=1e-10)


def test_gyroscopic_term_free_rigid_body():
    ground_body = RigidBody(
        mass=1.0, inertia=np.diag([1.0, 2.0, 3.0]), omega_body=(1.0, 2.0, 3.0)
    )
    layout = SystemCoordinates([ground_body])
    vdot, lam = _quasi.solve_accelerations_quasi(layout, [])
    # Euler equations: J w' = -w x J w
    w = np.array([1.0, 2.0, 3.0])
    J = np.diag([1.0, 2.0, 3.0])
    np.testing.assert_allclose(vdot[3:], np.linalg.solve(J, -np.cross(w, J @ w)))
    np.testing.assert_allclose(vdot[:3], 0.0)
    assert lam.size == 0


def test_redundant_constraints_do_not_break_solve():
    ground = Ground()
    body = RigidBody(mass=1.0, inertia=np.eye(3), position=(1.0, 0.0, 0.0))
    m_g = Marker(ground, position=(0.0, 0.0, 0.0))
    m_b = Marker(body, position=(-1.0, 0.0, 0.0))
    # Two identical revolute joints: every row is duplicated.
    joints = [RevoluteJoint(m_g, m_b), RevoluteJoint(m_g, m_b)]
    layout = SystemCoordinates([body])
    Qe = np.zeros(7)
    Qe[1] = -9.81
    vdot, lam = _quasi.solve_accelerations_quasi(layout, joints, Qe)
    assert np.all(np.isfinite(vdot)) and np.all(np.isfinite(lam))
    # pendulum released horizontally: alpha_z = -m g L / (J + m L^2)
    np.testing.assert_allclose(vdot[5], -9.81 / 2.0, atol=1e-9)
