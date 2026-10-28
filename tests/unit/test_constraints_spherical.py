"""Unit tests for ``modyn.constraints.spherical``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.constraints import SphericalJoint
from modyn.constraints.base import Constraint
from modyn.math import quaternions as _q
from modyn.model.body import RigidBody
from modyn.model.ground import Ground
from modyn.model.marker import Marker


@pytest.fixture(autouse=True)
def _reset_singletons():
    # Reset Ground singleton + per-class counters between tests.
    from modyn.core.base import Base

    Ground._instance = None
    Base.reset_all_counts()
    yield
    Ground._instance = None
    Base.reset_all_counts()


# ─── Helpers ──────────────────────────────────────────────────────────


def _make_body(
    *,
    R=(0.0, 0.0, 0.0),
    p=(1.0, 0.0, 0.0, 0.0),
    v=(0.0, 0.0, 0.0),
    w=(0.0, 0.0, 0.0),
) -> RigidBody:
    b = RigidBody(
        mass=1.0,
        inertia=np.eye(3),
        position=R,
        quaternion=p,
        velocity=v,
    )
    b.omega_global = np.asarray(w, dtype=float)
    return b


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def _phi_at(joint: SphericalJoint, body: RigidBody, R, p) -> np.ndarray:
    """Evaluate ``Phi`` after temporarily writing ``R, p`` into ``body``."""
    R0, p0 = body.position.copy(), body.quaternion.copy()
    body.position = np.asarray(R, dtype=float)
    body.quaternion = np.asarray(p, dtype=float)
    try:
        return joint.phi().copy()
    finally:
        body.position = R0
        body.quaternion = p0


# ─── SphericalJoint ───────────────────────────────────────────────────


class TestSphericalJoint:
    def test_is_a_constraint(self):
        g = Ground()
        b = _make_body()
        m_g = Marker(g)
        m_b = Marker(b)
        j = SphericalJoint(m_g, m_b)
        assert isinstance(j, Constraint)
        assert j.n_eq == 3

    def test_shapes(self):
        b1 = _make_body(R=(1.0, 0.0, 0.0))
        b2 = _make_body(R=(2.0, 0.0, 0.0))
        m1 = Marker(b1, position=(0.1, 0.0, 0.0))
        m2 = Marker(b2, position=(-0.2, 0.0, 0.0))
        j = SphericalJoint(m1, m2)

        assert j.phi().shape == (3,)
        blocks = j.phi_q()
        assert len(blocks) == 2
        for _body, J_R, J_p in blocks:
            assert J_R.shape == (3, 3)
            assert J_p.shape == (3, 4)
        assert j.gamma().shape == (3,)

    def test_residual_zero_when_coincident(self):
        # Two markers placed so their world origins coincide at (1,2,3).
        b1 = _make_body(R=(1.0, 2.0, 3.0))
        b2 = _make_body(R=(0.5, 1.5, 2.5))
        m1 = Marker(b1, position=(0.0, 0.0, 0.0))
        m2 = Marker(b2, position=(0.5, 0.5, 0.5))
        j = SphericalJoint(m1, m2)

        np.testing.assert_allclose(j.phi(), 0.0, atol=1e-15)

    def test_residual_value(self):
        b1 = _make_body(R=(1.0, 0.0, 0.0))
        b2 = _make_body(R=(0.0, 0.0, 0.0))
        m1 = Marker(b1, position=(0.0, 0.0, 0.0))
        m2 = Marker(b2, position=(0.0, 0.0, 0.0))
        j = SphericalJoint(m1, m2)
        np.testing.assert_allclose(j.phi(), [1.0, 0.0, 0.0])

    def test_ground_block_is_omitted(self):
        g = Ground()
        b = _make_body(R=(1.0, 0.0, 0.0))
        m_g = Marker(g, position=(0.0, 0.0, 0.0))
        m_b = Marker(b, position=(0.0, 0.0, 0.0))
        j = SphericalJoint(m_g, m_b)
        blocks = j.phi_q()
        assert len(blocks) == 1
        body, _J_R, _J_p = blocks[0]
        assert body is b

    def test_J_R_signs(self):
        b1 = _make_body()
        b2 = _make_body()
        m1 = Marker(b1)
        m2 = Marker(b2)
        j = SphericalJoint(m1, m2)
        blocks = {body: (J_R, J_p) for body, J_R, J_p in j.phi_q()}
        np.testing.assert_allclose(blocks[b1][0], np.eye(3))
        np.testing.assert_allclose(blocks[b2][0], -np.eye(3))

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_phi_q_finite_difference(self, seed):
        """``Phi_q`` agrees with a finite-difference Jacobian for both bodies."""
        rng = np.random.default_rng(seed)
        b1 = _make_body(R=rng.standard_normal(3), p=_random_quat(rng))
        b2 = _make_body(R=rng.standard_normal(3), p=_random_quat(rng))
        m1 = Marker(b1, position=rng.standard_normal(3))
        m2 = Marker(b2, position=rng.standard_normal(3))
        j = SphericalJoint(m1, m2)

        blocks = {body: (J_R, J_p) for body, J_R, J_p in j.phi_q()}
        eps = 1e-7
        for body in (b1, b2):
            J_R, J_p = blocks[body]
            R0, p0 = body.position.copy(), body.quaternion.copy()

            # ∂Φ/∂R via central differences.
            J_R_num = np.empty((3, 3))
            for k in range(3):
                e = np.zeros(3)
                e[k] = eps
                f_plus = _phi_at(j, body, R0 + e, p0)
                f_minus = _phi_at(j, body, R0 - e, p0)
                J_R_num[:, k] = (f_plus - f_minus) / (2 * eps)
            np.testing.assert_allclose(J_R, J_R_num, atol=1e-9)

            # ∂Φ/∂p via central differences. NOTE: the body's setter
            # re-normalises the quaternion, but our analytic Jacobian
            # is the unconstrained derivative — so we perturb along
            # *unnormalised* directions and divide by the resulting
            # finite difference of the normalised quaternion. The cleaner
            # equivalent is to compare the action on tangent vectors
            # (p^T δp = 0) only, where normalisation is a no-op to first
            # order. We use four tangent directions spanning the 3D
            # tangent space and check J_p · δp == numerical ΔΦ.
            # Build an orthonormal basis of the tangent space at p0.
            # Any 3 of the 4 columns of (I - p0 p0^T) work.
            T = np.eye(4) - np.outer(p0, p0)
            # Use 4 tangent perturbations.
            for k in range(4):
                dp = T[:, k]
                norm_dp = np.linalg.norm(dp)
                if norm_dp < 1e-12:
                    continue
                dp_unit = dp / norm_dp
                f_plus = _phi_at(j, body, R0, p0 + eps * dp_unit)
                f_minus = _phi_at(j, body, R0, p0 - eps * dp_unit)
                num = (f_plus - f_minus) / (2 * eps)
                ana = J_p @ dp_unit
                np.testing.assert_allclose(ana, num, atol=1e-7)

            # State restored.
            np.testing.assert_array_equal(body.position, R0)
            np.testing.assert_array_equal(body.quaternion, p0)

    @pytest.mark.parametrize("seed", [0, 3, 11])
    def test_gamma_matches_phi_ddot(self, seed):
        """At an arbitrary state and velocity, ``Phi_q q̈ = γ`` for q̈ = 0.

        Concretely: differentiating ``Φ(q(t))`` twice in time gives
        ``Φ̈ = Φ_q q̈ + Φ̇_q q̇``. With q̈ = 0 we must have ``Φ̇_q q̇ = -γ``,
        i.e. ``γ = -Φ̇_q q̇``, which is the design contract of ``gamma()``.
        We verify this by finite-differencing ``Φ_q q̇`` in time.
        """
        rng = np.random.default_rng(seed)
        b1 = _make_body(
            R=rng.standard_normal(3),
            p=_random_quat(rng),
            v=rng.standard_normal(3),
            w=rng.standard_normal(3),
        )
        b2 = _make_body(
            R=rng.standard_normal(3),
            p=_random_quat(rng),
            v=rng.standard_normal(3),
            w=rng.standard_normal(3),
        )
        m1 = Marker(b1, position=rng.standard_normal(3))
        m2 = Marker(b2, position=rng.standard_normal(3))
        j = SphericalJoint(m1, m2)

        gamma_analytic = j.gamma()

        # Compute Φ̇_q · q̇ analytically through finite differences of Φ in time.
        # Take a small Euler step in q with q̇ = (Ṙ, ṗ), evaluate ΔΦ / Δt, and
        # compute the centered second derivative.
        def step(body: RigidBody, dt: float) -> tuple[np.ndarray, np.ndarray]:
            R = body.position + dt * body.velocity
            pdot = _q.omega_to_pdot(body.quaternion, body.omega_global)
            p = body.quaternion + dt * pdot
            p = p / np.linalg.norm(p)
            return R, p

        dt = 1e-5
        # Phi at t = ±dt and 0.
        R1_p, p1_p = step(b1, +dt)
        R2_p, p2_p = step(b2, +dt)
        R1_m, p1_m = step(b1, -dt)
        R2_m, p2_m = step(b2, -dt)
        R1_0, p1_0 = b1.position.copy(), b1.quaternion.copy()
        R2_0, p2_0 = b2.position.copy(), b2.quaternion.copy()

        def set_state(R1, p1, R2, p2):
            b1.position = R1
            b1.quaternion = p1
            b2.position = R2
            b2.quaternion = p2

        set_state(R1_p, p1_p, R2_p, p2_p)
        phi_p = j.phi().copy()
        set_state(R1_m, p1_m, R2_m, p2_m)
        phi_m = j.phi().copy()
        set_state(R1_0, p1_0, R2_0, p2_0)
        phi_0 = j.phi().copy()

        phi_ddot_num = (phi_p - 2 * phi_0 + phi_m) / dt**2
        # Our trajectory has q̈ = 0 (constant-velocity advance in (R, p),
        # then renormalised — the normalisation introduces only O(dt²)
        # curvature on p). The DAE acceleration-level identity reads
        # Φ_q q̈ + Φ̇_q q̇ = 0, with γ ≡ -Φ̇_q q̇ so that Φ_q q̈ = γ. With
        # q̈ = 0 this gives Φ̈ = Φ̇_q q̇ = -γ. Hence:
        np.testing.assert_allclose(phi_ddot_num, -gamma_analytic, atol=1e-4)

    def test_marker_accessors(self):
        b1 = _make_body()
        b2 = _make_body()
        m1 = Marker(b1)
        m2 = Marker(b2)
        j = SphericalJoint(m1, m2, name="ball")
        assert j.marker_i is m1
        assert j.marker_j is m2
        assert j.name == "ball"
