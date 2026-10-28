r"""Equations of motion in quasi-velocities (the PMD-like formulation).

The velocity variables of every body are the translational velocity and the
angular velocity in the **body frame**,

.. math::

    \mathbf v_B = [\dot{\mathbf R},\ \boldsymbol\omega']^\mathsf{T} \in \mathbb R^6,

while the positions remain :math:`\mathbf q_B = [\mathbf R,\ \mathbf p]`. The
two sets are related by the kinematic map

.. math::

    \dot{\mathbf q} = \mathbf T(\mathbf p)\,\mathbf v, \qquad
    \mathbf T_B = \begin{bmatrix} \mathbf I_3 & \mathbf 0 \\
    \mathbf 0 & \tfrac12 G^\mathsf{T} \end{bmatrix},

which follows from :math:`\boldsymbol\omega' = 2G\dot{\mathbf p}` and
:math:`G G^\mathsf{T} = \mathbf I`. With these variables the mass matrix is
constant and block diagonal, exactly as in the planar formulation,

.. math::

    \mathbf M = \operatorname{diag}(m\,\mathbf I_3,\ \bar{\mathbf J}),

and the only new term is the gyroscopic force
:math:`-\boldsymbol\omega'\times\bar{\mathbf J}\boldsymbol\omega'`. The
constraint Jacobian in quasi-velocities is :math:`\mathbf D = \boldsymbol\Phi_q\mathbf T`, so that
:math:`\dot{\boldsymbol\Phi} = \mathbf D\mathbf v + \boldsymbol\Phi_t`, and the
equations of motion read

.. math::

    \mathbf M\dot{\mathbf v} = \mathbf h(\mathbf q,\mathbf v,t) - \mathbf D^\mathsf{T}\boldsymbol\lambda,
    \qquad \mathbf D\dot{\mathbf v} = \boldsymbol\gamma_v,

where :math:`\mathbf h = \mathbf T^\mathsf{T}\mathbf Q_e + \mathbf h_{gyro}`.
The Euler-parameter normalization :math:`\mathbf p^\mathsf{T}\mathbf p - 1 = 0`
does not appear in :math:`\mathbf D`: it is a constraint on the positions
only and is handled by the integrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..math import quaternions as _q
from . import eom as _eom

if TYPE_CHECKING:
    from ..constraints.base import Constraint
    from .coordinates import SystemCoordinates

__all__ = [
    "N_VEL_PER_BODY",
    "applied_to_quasi",
    "assemble_velocity",
    "constraint_jacobian_quasi",
    "constraint_rhs_quasi",
    "gyroscopic_forces",
    "kinematic_map",
    "mass_matrix_quasi",
    "n_velocities",
    "qdot_from_velocity",
    "scatter_velocity",
    "solve_accelerations_quasi",
    "velocity_from_qdot",
]

#: Velocity variables contributed by one rigid body: ``[Ṙ(3), ω'(3)]``.
N_VEL_PER_BODY: int = 6


def n_velocities(layout: SystemCoordinates) -> int:
    """Dimension of the quasi-velocity vector, ``6 · n_bodies``."""
    return N_VEL_PER_BODY * layout.n_bodies


def _vslice(k: int) -> slice:
    return slice(N_VEL_PER_BODY * k, N_VEL_PER_BODY * (k + 1))


# ─── Mass, kinematic map, gyroscopic forces ───────────────────────────


def mass_matrix_quasi(layout: SystemCoordinates) -> NDArray[np.float64]:
    r"""Constant block-diagonal mass matrix :math:`\operatorname{diag}(m\mathbf I_3, \bar{\mathbf J})`."""
    nv = n_velocities(layout)
    M = np.zeros((nv, nv), dtype=np.float64)
    for k, body in enumerate(layout.bodies):
        s = _vslice(k)
        M[s.start : s.start + 3, s.start : s.start + 3] = body.mass * np.eye(3)
        M[s.start + 3 : s.stop, s.start + 3 : s.stop] = body.inertia
    return M


def kinematic_map(layout: SystemCoordinates) -> NDArray[np.float64]:
    r"""Map :math:`\mathbf T(\mathbf p)` with :math:`\dot{\mathbf q} = \mathbf T\mathbf v`, shape ``(7n, 6n)``."""
    T = np.zeros((layout.n_coords, n_velocities(layout)), dtype=np.float64)
    for k, body in enumerate(layout.bodies):
        s = layout.offset(body)
        v = _vslice(k).start
        T[s : s + 3, v : v + 3] = np.eye(3)
        T[s + 3 : s + 7, v + 3 : v + 6] = 0.5 * _q.G(body.quaternion).T
    return T


def assemble_velocity(layout: SystemCoordinates) -> NDArray[np.float64]:
    r"""Gather :math:`\mathbf v = [\dot{\mathbf R}, \boldsymbol\omega']` from the bodies."""
    v = np.empty(n_velocities(layout), dtype=np.float64)
    for k, body in enumerate(layout.bodies):
        s = _vslice(k)
        v[s.start : s.start + 3] = body.velocity
        v[s.start + 3 : s.stop] = body.omega_body
    return v


def scatter_velocity(layout: SystemCoordinates, v: ArrayLike) -> None:
    """Distribute a quasi-velocity vector onto the bodies."""
    v_arr = np.asarray(v, dtype=np.float64).reshape(n_velocities(layout))
    for k, body in enumerate(layout.bodies):
        s = _vslice(k)
        body.velocity = v_arr[s.start : s.start + 3]
        body.omega_body = v_arr[s.start + 3 : s.stop]


def qdot_from_velocity(layout: SystemCoordinates, v: ArrayLike) -> NDArray[np.float64]:
    r"""Position-coordinate rates :math:`\dot{\mathbf q} = \mathbf T(\mathbf p)\mathbf v`."""
    return kinematic_map(layout) @ np.asarray(v, dtype=np.float64).reshape(-1)


def velocity_from_qdot(layout: SystemCoordinates, qdot: ArrayLike) -> NDArray[np.float64]:
    r"""Quasi-velocities from coordinate rates: :math:`\dot{\mathbf R}` and :math:`\boldsymbol\omega' = 2G\dot{\mathbf p}`."""
    qd = np.asarray(qdot, dtype=np.float64).reshape(layout.n_coords)
    v = np.empty(n_velocities(layout), dtype=np.float64)
    for k, body in enumerate(layout.bodies):
        s = layout.offset(body)
        vs = _vslice(k)
        v[vs.start : vs.start + 3] = qd[s : s + 3]
        v[vs.start + 3 : vs.stop] = _q.pdot_to_omega_body(body.quaternion, qd[s + 3 : s + 7])
    return v


def gyroscopic_forces(layout: SystemCoordinates) -> NDArray[np.float64]:
    r"""Gyroscopic term :math:`[\mathbf 0,\ -\boldsymbol\omega'\times\bar{\mathbf J}\boldsymbol\omega']` per body."""
    h = np.zeros(n_velocities(layout), dtype=np.float64)
    for k, body in enumerate(layout.bodies):
        s = _vslice(k)
        w = body.omega_body
        h[s.start + 3 : s.stop] = -np.cross(w, body.inertia @ w)
    return h


def applied_to_quasi(layout: SystemCoordinates, Qe: ArrayLike) -> NDArray[np.float64]:
    r"""Map generalized forces on :math:`\mathbf q` to forces on :math:`\mathbf v`.

    Since :math:`(\mathbf Q_e)_p = 2G^\mathsf{T}\boldsymbol\tau'`, the product
    :math:`\mathbf T^\mathsf{T}\mathbf Q_e` returns :math:`[\mathbf F, \boldsymbol\tau']`.
    """
    Qe_arr = np.asarray(Qe, dtype=np.float64).reshape(layout.n_coords)
    return kinematic_map(layout).T @ Qe_arr


# ─── Constraints ──────────────────────────────────────────────────────


def _n_joint_eqs(constraints: list[Constraint]) -> int:
    return sum(c.n_eq for c in constraints)


def constraint_jacobian_quasi(
    constraints: list[Constraint], layout: SystemCoordinates
) -> NDArray[np.float64]:
    r"""Jacobian :math:`\mathbf D = \boldsymbol\Phi_q\mathbf T` of the joint constraints, shape ``(m, 6n)``."""
    m = _n_joint_eqs(constraints)
    Phi_q = _eom.constraint_jacobian(constraints, layout)[:m]
    return Phi_q @ kinematic_map(layout)


def constraint_rhs_quasi(
    constraints: list[Constraint], layout: SystemCoordinates
) -> NDArray[np.float64]:
    r"""Acceleration-level right-hand side :math:`\boldsymbol\gamma_v` of :math:`\mathbf D\dot{\mathbf v} = \boldsymbol\gamma_v`.

    Starting from :math:`\boldsymbol\Phi_q\ddot{\mathbf q} = \boldsymbol\gamma` and
    :math:`\ddot{\mathbf q} = \mathbf T\dot{\mathbf v} + \dot{\mathbf T}\mathbf v`,
    :math:`\boldsymbol\gamma_v = \boldsymbol\gamma - \boldsymbol\Phi_q\dot{\mathbf T}\mathbf v`
    with :math:`\dot{\mathbf T}\mathbf v = [\mathbf 0,\ \tfrac12 G(\dot{\mathbf p})^\mathsf{T}\boldsymbol\omega']`.
    """
    m = _n_joint_eqs(constraints)
    gamma = _eom.constraint_rhs(constraints, layout)[:m]
    Phi_q = _eom.constraint_jacobian(constraints, layout)[:m]
    Tdot_v = np.zeros(layout.n_coords, dtype=np.float64)
    for body in layout.bodies:
        s = layout.offset(body)
        w = body.omega_body
        pdot = _q.omega_body_to_pdot(body.quaternion, w)
        Tdot_v[s + 3 : s + 7] = 0.5 * _q.G(pdot).T @ w
    return gamma - Phi_q @ Tdot_v


def solve_accelerations_quasi(
    layout: SystemCoordinates,
    constraints: list[Constraint],
    applied_forces: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Solve :math:`[\mathbf M, \mathbf D^\mathsf{T}; \mathbf D, \mathbf 0][\dot{\mathbf v}; \boldsymbol\lambda] = [\mathbf h; \boldsymbol\gamma_v]`.

    Parameters
    ----------
    layout, constraints
        System description; the state is read from the bodies.
    applied_forces
        Generalized forces :math:`\mathbf Q_e` on :math:`\mathbf q`, shape
        ``(7n,)`` (the convention of the force classes). Defaults to zero.

    Returns
    -------
    vdot, lam
        Quasi-accelerations and joint multipliers. When the constraints are
        redundant the minimum-norm multipliers are returned.
    """
    nv = n_velocities(layout)
    m = _n_joint_eqs(constraints)
    M = mass_matrix_quasi(layout)
    h = gyroscopic_forces(layout)
    if applied_forces is not None:
        h = h + applied_to_quasi(layout, applied_forces)
    if m == 0:
        return np.linalg.solve(M, h), np.zeros(0)

    D = constraint_jacobian_quasi(constraints, layout)
    gamma_v = constraint_rhs_quasi(constraints, layout)
    S = np.zeros((nv + m, nv + m), dtype=np.float64)
    S[:nv, :nv] = M
    S[:nv, nv:] = D.T
    S[nv:, :nv] = D
    rhs = np.concatenate((h, gamma_v))
    sol, *_ = np.linalg.lstsq(S, rhs, rcond=None)
    return sol[:nv], sol[nv:]
