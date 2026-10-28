"""Unit tests for ``modyn.solver.kinematics``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.constraints import RevoluteJoint, SphericalJoint
from modyn.dynamics import SystemCoordinates
from modyn.dynamics.eom import constraint_jacobian, constraint_residual
from modyn.model.body import RigidBody
from modyn.model.marker import Marker
from modyn.solver.kinematics import (
    AssemblyResult,
    assemble_position,
    solve_velocity,
)


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(4)
    return v / np.linalg.norm(v)


def _make_body(rng: np.random.Generator) -> RigidBody:
    return RigidBody(
        mass=1.0 + abs(rng.standard_normal()),
        inertia=np.diag(1.0 + rng.random(3)),
        position=rng.standard_normal(3),
        quaternion=_random_quat(rng),
        velocity=rng.standard_normal(3),
        omega_body=rng.standard_normal(3),
    )


class TestAssemblePosition:
    def test_already_assembled_converges_immediately(self):
        # Two coincident markers ⇒ spherical residual already zero.
        rng = np.random.default_rng(0)
        b1 = _make_body(rng)
        b2 = _make_body(rng)
        # Force coincident origins (markers at body centroids).
        b2.position = b1.position.copy()
        joint = SphericalJoint(Marker(b1), Marker(b2))
        layout = SystemCoordinates([b1, b2])
        result = assemble_position(layout, [joint])
        assert isinstance(result, AssemblyResult)
        assert result.converged
        assert result.iterations == 0
        assert result.residual_norm <= 1e-10

    def test_spherical_assembly_drives_residual_to_zero(self):
        rng = np.random.default_rng(1)
        b1 = _make_body(rng)
        b2 = _make_body(rng)
        joint = SphericalJoint(
            Marker(b1, position=rng.standard_normal(3)),
            Marker(b2, position=rng.standard_normal(3)),
        )
        layout = SystemCoordinates([b1, b2])
        result = assemble_position(layout, [joint], tol=1e-12)
        assert result.converged
        phi = constraint_residual([joint], layout)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_revolute_assembly(self):
        rng = np.random.default_rng(2)
        b1 = _make_body(rng)
        b2 = _make_body(rng)
        joint = RevoluteJoint(
            Marker(b1, position=rng.standard_normal(3)),
            Marker(b2, position=rng.standard_normal(3)),
        )
        layout = SystemCoordinates([b1, b2])
        result = assemble_position(layout, [joint], tol=1e-11)
        assert result.converged
        np.testing.assert_allclose(constraint_residual([joint], layout), 0.0, atol=1e-9)

    def test_quaternions_stay_unit_after_assembly(self):
        rng = np.random.default_rng(3)
        b1, b2 = _make_body(rng), _make_body(rng)
        joint = SphericalJoint(Marker(b1), Marker(b2))
        layout = SystemCoordinates([b1, b2])
        assemble_position(layout, [joint])
        for b in (b1, b2):
            np.testing.assert_allclose(np.linalg.norm(b.quaternion), 1.0, atol=1e-12)

    def test_invalid_arguments(self):
        rng = np.random.default_rng(4)
        b1, b2 = _make_body(rng), _make_body(rng)
        joint = SphericalJoint(Marker(b1), Marker(b2))
        layout = SystemCoordinates([b1, b2])
        with pytest.raises(ValueError):
            assemble_position(layout, [joint], tol=0.0)
        with pytest.raises(ValueError):
            assemble_position(layout, [joint], max_iter=0)


class TestSolveVelocity:
    def test_projection_satisfies_velocity_constraint(self):
        rng = np.random.default_rng(5)
        b1 = _make_body(rng)
        b2 = _make_body(rng)
        joint = SphericalJoint(
            Marker(b1, position=rng.standard_normal(3)),
            Marker(b2, position=rng.standard_normal(3)),
        )
        layout = SystemCoordinates([b1, b2])
        # Assemble first so the Jacobian is evaluated on the manifold.
        assemble_position(layout, [joint], tol=1e-12)
        qdot = solve_velocity(layout, [joint])
        Phi_q = constraint_jacobian([joint], layout)
        np.testing.assert_allclose(Phi_q @ qdot, 0.0, atol=1e-10)

    def test_quaternion_rate_is_tangent(self):
        rng = np.random.default_rng(6)
        b = _make_body(rng)
        # Single free body, only the normalization constraint applies.
        layout = SystemCoordinates([b])
        solve_velocity(layout, [])
        pdot = layout.assemble_qdot()[3:7]
        np.testing.assert_allclose(b.quaternion @ pdot, 0.0, atol=1e-12)

    def test_minimum_correction_leaves_consistent_velocity_unchanged(self):
        rng = np.random.default_rng(7)
        b1 = _make_body(rng)
        b2 = _make_body(rng)
        joint = SphericalJoint(
            Marker(b1, position=rng.standard_normal(3)),
            Marker(b2, position=rng.standard_normal(3)),
        )
        layout = SystemCoordinates([b1, b2])
        assemble_position(layout, [joint], tol=1e-12)
        qdot1 = solve_velocity(layout, [joint])
        # Applying the projection again must be (numerically) idempotent.
        qdot2 = solve_velocity(layout, [joint])
        np.testing.assert_allclose(qdot1, qdot2, atol=1e-10)

    def test_rhs_is_honored(self):
        rng = np.random.default_rng(8)
        b1 = _make_body(rng)
        b2 = _make_body(rng)
        joint = SphericalJoint(Marker(b1), Marker(b2))
        layout = SystemCoordinates([b1, b2])
        assemble_position(layout, [joint], tol=1e-12)
        Phi_q = constraint_jacobian([joint], layout)
        nu = rng.standard_normal(Phi_q.shape[0])
        # Make the rhs reachable: project it onto the row space so lstsq is exact.
        nu = Phi_q @ np.linalg.lstsq(Phi_q, nu, rcond=None)[0]
        qdot = solve_velocity(layout, [joint], rhs=nu)
        np.testing.assert_allclose(constraint_jacobian([joint], layout) @ qdot, nu, atol=1e-9)
