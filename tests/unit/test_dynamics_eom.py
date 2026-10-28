"""Unit tests for ``modyn.dynamics.eom``."""

from __future__ import annotations

import numpy as np

from modyn.constraints import SphericalJoint
from modyn.dynamics import SystemCoordinates
from modyn.dynamics import eom as _eom
from modyn.math import quaternions as _q
from modyn.model.body import RigidBody
from modyn.model.ground import Ground
from modyn.model.marker import Marker


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(4)
    return v / np.linalg.norm(v)


def _make_body(rng: np.random.Generator, *, omega=None) -> RigidBody:
    return RigidBody(
        mass=1.0 + abs(rng.standard_normal()),
        inertia=np.diag(1.0 + rng.random(3)),
        position=rng.standard_normal(3),
        quaternion=_random_quat(rng),
        velocity=rng.standard_normal(3),
        omega_body=rng.standard_normal(3) if omega is None else np.asarray(omega, float),
    )


class TestMassMatrix:
    def test_shape_and_block_structure(self):
        rng = np.random.default_rng(0)
        bodies = [_make_body(rng) for _ in range(2)]
        layout = SystemCoordinates(bodies)
        M = _eom.mass_matrix(layout)
        assert M.shape == (14, 14)
        # Translational block = m I3.
        for b in bodies:
            s = layout.offset(b)
            np.testing.assert_allclose(M[s : s + 3, s : s + 3], b.mass * np.eye(3))

    def test_symmetric_and_psd(self):
        rng = np.random.default_rng(1)
        layout = SystemCoordinates([_make_body(rng) for _ in range(3)])
        M = _eom.mass_matrix(layout)
        np.testing.assert_allclose(M, M.T, atol=1e-14)
        eigs = np.linalg.eigvalsh(M)
        assert eigs.min() > -1e-10  # PSD (p-block is singular along p)

    def test_p_block_annihilates_quaternion(self):
        # 4 G^T J G p = 0 since G p = 0.
        rng = np.random.default_rng(2)
        b = _make_body(rng)
        layout = SystemCoordinates([b])
        M = _eom.mass_matrix(layout)
        Mpp = M[3:7, 3:7]
        np.testing.assert_allclose(Mpp @ b.quaternion, 0.0, atol=1e-13)


class TestQuadraticVelocity:
    def test_zero_when_no_angular_velocity(self):
        rng = np.random.default_rng(3)
        b = _make_body(rng, omega=(0.0, 0.0, 0.0))
        layout = SystemCoordinates([b])
        Qv = _eom.quadratic_velocity_forces(layout)
        np.testing.assert_allclose(Qv, 0.0, atol=1e-14)

    def test_translational_part_is_zero(self):
        rng = np.random.default_rng(4)
        bodies = [_make_body(rng) for _ in range(2)]
        layout = SystemCoordinates(bodies)
        Qv = _eom.quadratic_velocity_forces(layout)
        for b in bodies:
            s = layout.offset(b)
            np.testing.assert_allclose(Qv[s : s + 3], 0.0, atol=1e-14)

    def test_matches_newton_euler_torque(self):
        # For a body in free motion the generalized p-force must reproduce the
        # body-frame Euler equation: premultiplying M_pp p̈ = Q_v,p by the
        # pseudo-inverse mapping gives ω̇' such that J ω̇' + ω'×Jω' = 0.
        rng = np.random.default_rng(5)
        b = _make_body(rng)
        layout = SystemCoordinates([b])
        M = _eom.mass_matrix(layout)
        Qv = _eom.quadratic_velocity_forces(layout)
        p, w, J = b.quaternion, b.omega_body, b.inertia
        # Solve the (regularized) p-equation 4G^TJG p̈ = Q_v,p with the
        # normalization 2 p^T p̈ = -2 ṗ^T ṗ to remove the null direction.
        pdot = _q.omega_body_to_pdot(p, w)
        Mpp = M[3:7, 3:7]
        A = np.zeros((5, 4))
        A[:4] = Mpp
        A[4] = 2.0 * p
        rhs = np.empty(5)
        rhs[:4] = Qv[3:7]
        rhs[4] = -2.0 * (pdot @ pdot)
        pddot, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        # Recover ω̇' = 2 G p̈ + 2 Ġ ṗ and check Euler equation residual.
        Gdot = _q.G(pdot)
        wdot = 2.0 * _q.G(p) @ pddot + 2.0 * Gdot @ pdot
        residual = J @ wdot + np.cross(w, J @ w)
        np.testing.assert_allclose(residual, 0.0, atol=1e-8)


class TestConstraintAssembly:
    def test_jacobian_shape_and_normalization_rows(self):
        rng = np.random.default_rng(6)
        b1 = _make_body(rng)
        b2 = _make_body(rng)
        joint = SphericalJoint(Marker(b1), Marker(b2))
        layout = SystemCoordinates([b1, b2])
        Phi_q = _eom.constraint_jacobian([joint], layout)
        # 3 joint rows + 2 normalization rows, 14 columns.
        assert Phi_q.shape == (5, 14)
        # Normalization rows: p-block = 2 p, R-block = 0.
        for k, b in enumerate((b1, b2)):
            s = layout.offset(b)
            row = 3 + k
            np.testing.assert_allclose(Phi_q[row, s : s + 3], 0.0)
            np.testing.assert_allclose(Phi_q[row, s + 3 : s + 7], 2.0 * b.quaternion)

    def test_residual_normalization_zero_for_unit_quaternions(self):
        rng = np.random.default_rng(7)
        b1, b2 = _make_body(rng), _make_body(rng)
        joint = SphericalJoint(Marker(b1), Marker(b2))
        layout = SystemCoordinates([b1, b2])
        phi = _eom.constraint_residual([joint], layout)
        assert phi.shape == (5,)
        np.testing.assert_allclose(phi[3:], 0.0, atol=1e-14)

    def test_gamma_normalization(self):
        rng = np.random.default_rng(8)
        b = _make_body(rng)
        layout = SystemCoordinates([b])
        gamma = _eom.constraint_rhs([], layout)
        pdot = _q.omega_body_to_pdot(b.quaternion, b.omega_body)
        np.testing.assert_allclose(gamma[-1], -2.0 * (pdot @ pdot))

    def test_jacobian_skips_ground(self):
        Ground._instance = None
        g = Ground()
        b = _make_body(np.random.default_rng(9))
        joint = SphericalJoint(Marker(g), Marker(b))
        layout = SystemCoordinates([b])
        Phi_q = _eom.constraint_jacobian([joint], layout)
        # 3 joint rows + 1 normalization, 7 columns (only the free body).
        assert Phi_q.shape == (4, 7)


class TestSolve:
    def test_free_body_falls_under_gravity(self):
        # Single free body, no joints, gravity along -z, at rest rotationally.
        b = RigidBody(
            mass=2.0,
            inertia=np.diag([1.0, 2.0, 3.0]),
            position=(0.0, 0.0, 5.0),
            quaternion=(1.0, 0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            omega_body=(0.0, 0.0, 0.0),
        )
        layout = SystemCoordinates([b])
        g = -9.81
        Qe = np.zeros(7)
        Qe[2] = b.mass * g  # force = m g on the z translational coordinate
        qddot, lam = _eom.solve_accelerations(layout, [], Qe)
        np.testing.assert_allclose(qddot[:3], (0.0, 0.0, g), atol=1e-12)
        np.testing.assert_allclose(qddot[3:7], 0.0, atol=1e-12)
        np.testing.assert_allclose(lam, 0.0, atol=1e-12)

    def test_solution_satisfies_acceleration_constraint(self):
        rng = np.random.default_rng(10)
        b1 = _make_body(rng)
        b2 = _make_body(rng)
        # Place the two markers so the joint is consistent is not required for
        # the acceleration-level identity Φ_q q̈ = γ to hold by construction.
        joint = SphericalJoint(
            Marker(b1, position=rng.standard_normal(3)), Marker(b2, position=rng.standard_normal(3))
        )
        layout = SystemCoordinates([b1, b2])
        Qe = rng.standard_normal(layout.n_coords)
        qddot, lam = _eom.solve_accelerations(layout, [joint], Qe)
        Phi_q = _eom.constraint_jacobian([joint], layout)
        gamma = _eom.constraint_rhs([joint], layout)
        np.testing.assert_allclose(Phi_q @ qddot, gamma, atol=1e-9)
        # And the dynamic equation M q̈ + Φ_qᵀ λ = Qe + Qv.
        M = _eom.mass_matrix(layout)
        Qv = _eom.quadratic_velocity_forces(layout)
        np.testing.assert_allclose(M @ qddot + Phi_q.T @ lam, Qe + Qv, atol=1e-9)

    def test_assemble_augmented_block_sizes(self):
        rng = np.random.default_rng(11)
        b1, b2 = _make_body(rng), _make_body(rng)
        joint = SphericalJoint(Marker(b1), Marker(b2))
        layout = SystemCoordinates([b1, b2])
        sysm = _eom.assemble_augmented(layout, [joint])
        assert sysm.n_coords == 14
        assert sysm.n_eqs == 5
        assert sysm.A.shape == (19, 19)
        assert sysm.b.shape == (19,)
        # Saddle-point zero block.
        np.testing.assert_allclose(sysm.A[14:, 14:], 0.0)
