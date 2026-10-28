"""Shared helpers for joint unit tests.

Provides body/quaternion factories and finite-difference assertions used
by all joint test modules. Not a pytest module (leading underscore).
"""

from __future__ import annotations

import numpy as np

from modyn.math import quaternions as _q
from modyn.model.body import RigidBody

__all__ = [
    "assert_gamma_matches_phi_ddot",
    "assert_phi_q_matches_fd",
    "make_body",
    "phi_at",
    "random_quat",
]


def make_body(
    *,
    R=(0.0, 0.0, 0.0),
    p=(1.0, 0.0, 0.0, 0.0),
    v=(0.0, 0.0, 0.0),
    w=(0.0, 0.0, 0.0),
) -> RigidBody:
    """Build a unit-mass / unit-inertia rigid body with the given state."""
    b = RigidBody(
        mass=1.0,
        inertia=np.eye(3),
        position=R,
        quaternion=p,
        velocity=v,
    )
    b.omega_global = np.asarray(w, dtype=float)
    return b


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly-random unit quaternion."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def phi_at(joint, body: RigidBody, R, p) -> np.ndarray:
    """Evaluate ``joint.phi()`` after temporarily setting ``body``'s state."""
    R0, p0 = body.position.copy(), body.quaternion.copy()
    body.position = np.asarray(R, dtype=float)
    body.quaternion = np.asarray(p, dtype=float)
    try:
        return joint.phi().copy()
    finally:
        body.position = R0
        body.quaternion = p0


def assert_phi_q_matches_fd(
    joint,
    bodies,
    *,
    eps: float = 1e-7,
    atol_R: float = 1e-9,
    atol_p: float = 1e-7,
) -> None:
    r"""Central-difference check of ``phi_q`` against ``phi``.

    Checks :math:`\partial\boldsymbol\Phi/\partial \mathbf R` (Cartesian
    perturbations) and the tangent-quaternion projection of
    :math:`\partial\boldsymbol\Phi/\partial \mathbf p` (renormalisation is
    second-order so identity-tangent perturbations are safe).
    """
    blocks = {body: (J_R, J_p) for body, J_R, J_p in joint.phi_q()}
    for body in bodies:
        if body not in blocks:
            continue
        J_R, J_p = blocks[body]
        n_eq = J_R.shape[0]
        R0, p0 = body.position.copy(), body.quaternion.copy()

        # ∂Φ/∂R via central differences.
        J_R_num = np.empty((n_eq, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = eps
            f_plus = phi_at(joint, body, R0 + e, p0)
            f_minus = phi_at(joint, body, R0 - e, p0)
            J_R_num[:, k] = (f_plus - f_minus) / (2 * eps)
        np.testing.assert_allclose(J_R, J_R_num, atol=atol_R)

        # ∂Φ/∂p along the tangent of the unit-quaternion sphere.
        T = np.eye(4) - np.outer(p0, p0)
        for k in range(4):
            dp = T[:, k]
            norm_dp = np.linalg.norm(dp)
            if norm_dp < 1e-12:
                continue
            dp_unit = dp / norm_dp
            f_plus = phi_at(joint, body, R0, p0 + eps * dp_unit)
            f_minus = phi_at(joint, body, R0, p0 - eps * dp_unit)
            num = (f_plus - f_minus) / (2 * eps)
            ana = J_p @ dp_unit
            np.testing.assert_allclose(ana, num, atol=atol_p)

        np.testing.assert_array_equal(body.position, R0)
        np.testing.assert_array_equal(body.quaternion, p0)


def assert_gamma_matches_phi_ddot(
    joint,
    bodies,
    *,
    dt: float = 1e-5,
    atol: float = 1e-4,
) -> None:
    r"""Verify :math:`\boldsymbol\Phi_q\,\ddot{\mathbf q} = \boldsymbol\gamma`
    by integrating along a constant-velocity trajectory (so :math:`\ddot{\mathbf q}=0`)
    and checking :math:`\ddot{\boldsymbol\Phi}_\text{num} \approx -\boldsymbol\gamma`.
    """
    gamma_analytic = joint.gamma()
    state0 = {b: (b.position.copy(), b.quaternion.copy()) for b in bodies}

    def step(dt_signed: float) -> None:
        for b in bodies:
            R0, p0 = state0[b]
            R_new = R0 + dt_signed * b.velocity
            pdot = _q.omega_to_pdot(p0, b.omega_global)
            p_new = p0 + dt_signed * pdot
            p_new = p_new / np.linalg.norm(p_new)
            b.position = R_new
            b.quaternion = p_new

    def restore() -> None:
        for b in bodies:
            R0, p0 = state0[b]
            b.position = R0
            b.quaternion = p0

    restore()
    phi_0 = joint.phi().copy()
    step(+dt)
    phi_p = joint.phi().copy()
    step(-dt)
    phi_m = joint.phi().copy()
    restore()

    phi_ddot_num = (phi_p - 2 * phi_0 + phi_m) / dt**2
    np.testing.assert_allclose(phi_ddot_num, -gamma_analytic, atol=atol)
