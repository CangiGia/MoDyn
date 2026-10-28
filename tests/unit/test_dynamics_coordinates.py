"""Unit tests for ``modyn.dynamics.coordinates``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.dynamics import SystemCoordinates
from modyn.dynamics.coordinates import N_PER_BODY
from modyn.math import quaternions as _q
from modyn.model.body import RigidBody
from modyn.model.ground import Ground


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


class TestLayout:
    def test_dimensions(self):
        rng = np.random.default_rng(0)
        bodies = [_make_body(rng) for _ in range(3)]
        sc = SystemCoordinates(bodies)
        assert sc.n_bodies == 3
        assert sc.n_coords == 3 * N_PER_BODY == 21
        assert len(sc) == 3
        assert sc.bodies == tuple(bodies)

    def test_offsets_and_slices_are_contiguous(self):
        rng = np.random.default_rng(1)
        bodies = [_make_body(rng) for _ in range(4)]
        sc = SystemCoordinates(bodies)
        for k, b in enumerate(bodies):
            assert sc.offset(b) == k * N_PER_BODY
            assert sc.slice(b) == slice(k * N_PER_BODY, (k + 1) * N_PER_BODY)

    def test_order_is_preserved(self):
        rng = np.random.default_rng(2)
        b0, b1 = _make_body(rng), _make_body(rng)
        sc = SystemCoordinates([b1, b0])
        assert sc.offset(b1) == 0
        assert sc.offset(b0) == N_PER_BODY

    def test_unknown_body_raises_keyerror(self):
        rng = np.random.default_rng(3)
        bodies = [_make_body(rng) for _ in range(2)]
        stranger = _make_body(rng)
        sc = SystemCoordinates(bodies)
        with pytest.raises(KeyError):
            sc.offset(stranger)


class TestConstructionErrors:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one body"):
            SystemCoordinates([])

    def test_ground_raises(self):
        rng = np.random.default_rng(4)
        with pytest.raises(ValueError, match="Ground"):
            SystemCoordinates([_make_body(rng), Ground()])

    def test_non_rigidbody_raises(self):
        rng = np.random.default_rng(5)
        with pytest.raises(TypeError, match="expected RigidBody"):
            SystemCoordinates([_make_body(rng), "not a body"])

    def test_duplicate_raises(self):
        rng = np.random.default_rng(6)
        b = _make_body(rng)
        with pytest.raises(ValueError, match="more than once"):
            SystemCoordinates([b, b])


class TestGather:
    def test_assemble_q_matches_stacked_body_q(self):
        rng = np.random.default_rng(7)
        bodies = [_make_body(rng) for _ in range(3)]
        sc = SystemCoordinates(bodies)
        q = sc.assemble_q()
        assert q.shape == (sc.n_coords,)
        expected = np.concatenate([b.q for b in bodies])
        np.testing.assert_allclose(q, expected)

    def test_assemble_qdot_linear_and_quaternion_parts(self):
        rng = np.random.default_rng(8)
        bodies = [_make_body(rng) for _ in range(2)]
        sc = SystemCoordinates(bodies)
        qd = sc.assemble_qdot()
        assert qd.shape == (sc.n_coords,)
        for k, b in enumerate(bodies):
            blk = qd[k * N_PER_BODY : (k + 1) * N_PER_BODY]
            np.testing.assert_allclose(blk[:3], b.velocity)
            np.testing.assert_allclose(blk[3:], _q.omega_body_to_pdot(b.quaternion, b.omega_body))


class TestScatter:
    def test_scatter_q_sets_positions_and_normalizes_quaternion(self):
        rng = np.random.default_rng(9)
        bodies = [_make_body(rng) for _ in range(2)]
        sc = SystemCoordinates(bodies)
        q = rng.standard_normal(sc.n_coords)
        sc.scatter_q(q)
        for k, b in enumerate(bodies):
            blk = q[k * N_PER_BODY : (k + 1) * N_PER_BODY]
            np.testing.assert_allclose(b.position, blk[:3])
            np.testing.assert_allclose(np.linalg.norm(b.quaternion), 1.0)
            # Direction preserved after renormalization.
            np.testing.assert_allclose(b.quaternion, blk[3:] / np.linalg.norm(blk[3:]))

    def test_scatter_wrong_size_raises(self):
        rng = np.random.default_rng(10)
        sc = SystemCoordinates([_make_body(rng)])
        with pytest.raises(ValueError):
            sc.scatter_q(np.zeros(6))
        with pytest.raises(ValueError):
            sc.scatter_qdot(np.zeros(6))


class TestRoundTrip:
    def test_q_round_trip(self):
        rng = np.random.default_rng(11)
        bodies = [_make_body(rng) for _ in range(3)]
        sc = SystemCoordinates(bodies)
        q0 = sc.assemble_q()
        sc.scatter_q(q0)
        np.testing.assert_allclose(sc.assemble_q(), q0, atol=1e-15)

    def test_qdot_round_trip(self):
        rng = np.random.default_rng(12)
        bodies = [_make_body(rng) for _ in range(3)]
        sc = SystemCoordinates(bodies)
        qd0 = sc.assemble_qdot()
        sc.scatter_qdot(qd0)
        np.testing.assert_allclose(sc.assemble_qdot(), qd0, atol=1e-14)

    def test_set_state_then_assemble(self):
        rng = np.random.default_rng(13)
        bodies = [_make_body(rng) for _ in range(2)]
        sc = SystemCoordinates(bodies)
        # Build a consistent target state: q with unit quaternions, qdot with
        # quaternion rate already on the tangent space.
        target_q = sc.assemble_q()
        target_qd = sc.assemble_qdot()
        # Perturb then restore through set_state.
        sc.scatter_q(rng.standard_normal(sc.n_coords))
        sc.set_state(target_q, target_qd)
        np.testing.assert_allclose(sc.assemble_q(), target_q, atol=1e-15)
        np.testing.assert_allclose(sc.assemble_qdot(), target_qd, atol=1e-14)

    def test_scatter_qdot_preserves_omega_body(self):
        rng = np.random.default_rng(14)
        b = _make_body(rng)
        sc = SystemCoordinates([b])
        w0 = b.omega_body.copy()
        qd = sc.assemble_qdot()
        # Disturb then re-scatter: omega_body must come back exactly.
        b.omega_body = rng.standard_normal(3)
        sc.scatter_qdot(qd)
        np.testing.assert_allclose(b.omega_body, w0, atol=1e-14)
