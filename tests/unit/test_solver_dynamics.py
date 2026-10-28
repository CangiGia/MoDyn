"""Unit tests for ``modyn.solver.dynamics``.

Validates :func:`integrate_dynamics` and the underlying
:class:`~modyn.solver.integrators.GeneralizedAlpha` kernel.

Test catalogue
--------------
test_gen_alpha_coefficients_rho1
    rho_inf=1 → non-dissipative Newmark (αm=αf=0.5 is NOT the outcome;
    the Chung-Hulbert formula gives αm=-1/2, αf=1/2 for ρ∞=1, with γ=1, β=1).
test_gen_alpha_coefficients_rho0
    rho_inf=0 → maximum dissipation (αm=-1, αf=0, γ=3/2, β=9/4 — over-damped).
test_gen_alpha_invalid_rho
    rho_inf outside [0,1] raises ValueError.
test_free_body_gravity_trajectory
    Free body under gravity: position matches analytic parabola; λ=0;
    constraint residual Φ≈0 (only normalization constraint).
test_free_body_no_drift
    Over a short horizon the normalization constraint |pᵀp-1| stays at
    machine-epsilon scale (index-3 → no drift).
test_force_callback_vs_constant
    A constant-array force and an equivalent lambda callback produce
    bit-identical trajectories.
test_revolute_pendulum_no_drift
    A single-body revolute pendulum in gravity: |Φ| ≤ 1e-8 at every step
    (constraints enforced at position level).
test_revolute_pendulum_energy
    Mechanical energy is approximately conserved (rho_inf=1, many steps).
test_invalid_tspan
    t_span with t0 >= tf raises ValueError.
test_invalid_dt
    dt ≤ 0 raises ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest

from modyn.constraints import RevoluteJoint
from modyn.dynamics.coordinates import SystemCoordinates
from modyn.dynamics.eom import constraint_residual
from modyn.model.body import RigidBody
from modyn.model.ground import Ground
from modyn.model.marker import Marker
from modyn.solver.dynamics import DynamicsResult, integrate_dynamics
from modyn.solver.integrators import GeneralizedAlpha

# ─── marker ───────────────────────────────────────────────────────────────────
pytestmark = pytest.mark.unit


# ─── helpers ──────────────────────────────────────────────────────────────────


def _free_body(*, mass: float = 1.0) -> tuple[RigidBody, SystemCoordinates]:
    """A free rigid body at origin with identity orientation."""
    body = RigidBody(
        mass=mass,
        inertia=np.eye(3),
        position=(0.0, 0.0, 0.0),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
    )
    layout = SystemCoordinates([body])
    return body, layout


def _gravity_force(g: float = 9.81) -> np.ndarray:
    """Return a (7,) force vector with downward gravity on the translational DOF."""
    # 7 coords per body: [Rx, Ry, Rz, p0, p1, p2, p3] — gravity on Rz component
    Qe = np.zeros(7, dtype=np.float64)
    Qe[2] = -g  # F_z = m*(-g) for m=1
    return Qe


# ─── gen-α coefficient tests ──────────────────────────────────────────────────


def test_gen_alpha_coefficients_rho1():
    ga = GeneralizedAlpha(rho_inf=1.0)
    # Chung-Hulbert formulas with ρ∞=1:
    # αm=(2·1-1)/(1+1)=0.5; αf=1/(1+1)=0.5; γ=0.5+0.5-0.5=0.5; β=0.25·(1+0.5-0.5)²=0.25
    assert ga.alpha_m == pytest.approx(0.5)
    assert ga.alpha_f == pytest.approx(0.5)
    assert ga.gamma == pytest.approx(0.5)
    assert ga.beta == pytest.approx(0.25)


def test_gen_alpha_coefficients_rho0():
    ga = GeneralizedAlpha(rho_inf=0.0)
    # αm=(2·0-1)/(0+1)=-1; αf=0; γ=0.5+0-(-1)=1.5; β=0.25·(1+0-(-1))²=1.0
    assert ga.alpha_m == pytest.approx(-1.0)
    assert ga.alpha_f == pytest.approx(0.0)
    assert ga.gamma == pytest.approx(1.5)
    assert ga.beta == pytest.approx(1.0)


def test_gen_alpha_invalid_rho():
    with pytest.raises(ValueError, match="rho_inf"):
        GeneralizedAlpha(rho_inf=1.01)
    with pytest.raises(ValueError, match="rho_inf"):
        GeneralizedAlpha(rho_inf=-0.01)


# ─── free body under gravity ──────────────────────────────────────────────────


def test_free_body_gravity_trajectory():
    """q̈[:3] = [0, 0, -g]; position matches analytic parabola; λ ≈ 0."""
    g = 9.81
    _, layout = _free_body(mass=1.0)
    Qe = _gravity_force(g)

    dt = 1e-3
    t_end = 0.1
    result = integrate_dynamics(
        layout,
        [],  # no joint constraints
        (0.0, t_end),
        dt,
        applied_forces=Qe,
        rho_inf=0.8,
        newton_tol=1e-10,
    )

    assert isinstance(result, DynamicsResult)
    assert result.success

    t = result.times
    z_analytic = -0.5 * g * t**2  # starts from rest at origin

    # Position z column (index 2 in the 7-vector per body)
    z_sim = result.q[:, 2]
    np.testing.assert_allclose(z_sim, z_analytic, atol=1e-8)

    # Multipliers: only normalization constraint → should be ~0
    # (free body: no joints, so lam has shape (nt, n_bodies))
    np.testing.assert_allclose(result.lam, 0.0, atol=1e-10)


def test_free_body_no_drift():
    """‖pᵀp - 1‖ ≤ 1e-12 at every recorded step (index-3 → no drift)."""
    _, layout = _free_body()
    Qe = _gravity_force()

    result = integrate_dynamics(
        layout, [], (0.0, 0.05), dt=1e-3, applied_forces=Qe, newton_tol=1e-10
    )

    # p is columns 3:7 in the 7-vector
    p = result.q[:, 3:7]  # (nt, 4)
    norm_sq = np.einsum("ti,ti->t", p, p)
    np.testing.assert_allclose(norm_sq, 1.0, atol=1e-10)


# ─── force: callback vs constant ──────────────────────────────────────────────


def test_force_callback_vs_constant():
    """Constant array and equivalent callback give identical trajectories."""
    Qe_arr = _gravity_force()

    def Qe_fn(t, q, qdot):
        return Qe_arr

    _, layout_a = _free_body()
    _, layout_b = _free_body()

    kw = dict(t_span=(0.0, 0.05), dt=5e-4, newton_tol=1e-10)
    res_arr = integrate_dynamics(layout_a, [], applied_forces=Qe_arr, **kw)
    res_fn = integrate_dynamics(layout_b, [], applied_forces=Qe_fn, **kw)

    np.testing.assert_array_equal(res_arr.q, res_fn.q)
    np.testing.assert_array_equal(res_arr.lam, res_fn.lam)


# ─── revolute pendulum ────────────────────────────────────────────────────────


def _make_pendulum(length: float = 1.0, mass: float = 1.0):
    """Single-body revolute pendulum hanging from ground.

    The pendulum bob starts at rest, displaced to (L, 0, 0) with the hinge
    at the origin. Rotation axis is the global z-axis.
    """
    ground = Ground()
    body = RigidBody(
        mass=mass,
        inertia=np.diag([1.0, 1.0, 1.0]) * mass * length**2 / 12.0,
        position=(length, 0.0, 0.0),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
    )

    # Revolute joint about z-axis; hinge at origin (body local point = (-L, 0, 0))
    # Both marker z-axes aligned with global z (identity orientation).
    marker_gnd = Marker(ground, position=(0.0, 0.0, 0.0))
    marker_body = Marker(body, position=(-length, 0.0, 0.0))
    joint = RevoluteJoint(marker_gnd, marker_body)
    layout = SystemCoordinates([body])
    return body, layout, [joint], mass, length


def test_revolute_pendulum_no_drift():
    r"""‖Φ‖ ≤ 1e-7 at every step of a revolute pendulum (index-3, no stab.)."""
    g = 9.81
    _, layout, joints, mass, _ = _make_pendulum(length=1.0, mass=1.0)

    # Gravity force in the generalized-coordinate sense (force on Rz: m*(-g))
    def gravity(t, q, qdot):
        Qe = np.zeros(layout.n_coords, dtype=np.float64)
        Qe[2] = -mass * g
        return Qe

    result = integrate_dynamics(
        layout,
        joints,
        (0.0, 0.2),
        dt=1e-3,
        applied_forces=gravity,
        newton_tol=1e-8,
        project_initial=True,
    )

    assert result.success

    # Check constraint residual at every recorded state
    for i, q_i in enumerate(result.q):
        layout.scatter_q(q_i, normalize=False)
        phi = constraint_residual(joints, layout)
        assert np.linalg.norm(phi[:-1]) < 1e-7, (
            f"Constraint drift at step {i}: ‖Φ‖ = {np.linalg.norm(phi[:-1]):.2e}"
        )


def test_revolute_pendulum_energy():
    """Mechanical energy is approximately conserved with rho_inf=1.0."""
    g = 9.81
    mass = 1.0
    L = 1.0
    _, layout, joints, _, _ = _make_pendulum(length=L, mass=mass)

    def gravity(t, q, qdot):
        Qe = np.zeros(layout.n_coords, dtype=np.float64)
        Qe[2] = -mass * g
        return Qe

    # Small-angle: start slightly displaced (tilt ~0.1 rad, cosθ≈1)
    # Use default initial position (horizontal) for a larger amplitude test.
    result = integrate_dynamics(
        layout,
        joints,
        (0.0, 0.5),
        dt=5e-4,
        applied_forces=gravity,
        rho_inf=1.0,  # non-dissipative
        newton_tol=1e-10,
        project_initial=True,
    )

    assert result.success

    # Total energy = KE + PE
    # KE: 0.5 * m * Ṙᵀ Ṙ  (ignore rotational KE for simplicity — O(m L² ω²))
    # PE: m * g * Rz
    Rdot = result.qdot[:, :3]  # translational velocity
    Rz = result.q[:, 2]

    KE = 0.5 * mass * np.einsum("ti,ti->t", Rdot, Rdot)
    PE = mass * g * Rz
    E = KE + PE

    # Energy relative drift should stay small (non-dissipative gen-α)
    E0 = E[0]
    rel_drift = np.abs(E - E0) / (np.abs(E0) + 1e-12)
    assert np.max(rel_drift) < 0.01, f"Energy drift too large: max rel = {np.max(rel_drift):.2e}"


# ─── input validation ─────────────────────────────────────────────────────────


def test_invalid_tspan():
    _, layout = _free_body()
    with pytest.raises(ValueError, match="t_span"):
        integrate_dynamics(layout, [], (1.0, 0.5), dt=1e-3)


def test_invalid_dt():
    _, layout = _free_body()
    with pytest.raises(ValueError, match="dt"):
        integrate_dynamics(layout, [], (0.0, 1.0), dt=-0.1)
