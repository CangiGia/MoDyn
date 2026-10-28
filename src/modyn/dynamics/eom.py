r"""Augmented DAE assembly.

Builds, for the current state ``(q, q̇, t)`` stored on the bodies:

.. math::

    \begin{bmatrix} \mathbf M & \boldsymbol\Phi_q^\mathsf{T} \\
    \boldsymbol\Phi_q & \mathbf 0 \end{bmatrix}
    \begin{bmatrix} \ddot{\mathbf q} \\ \boldsymbol\lambda \end{bmatrix}
    = \begin{bmatrix} \mathbf Q_e + \mathbf Q_v \\ \boldsymbol\gamma \end{bmatrix}

where the constraint vector :math:`\boldsymbol\Phi` includes both the joint
constraints and the per-body Euler-parameter normalization
:math:`\mathbf p^\mathsf{T}\mathbf p - 1 = 0` (appended automatically here).

Per body the generalized coordinates are :math:`\mathbf q_B = [\mathbf R,
\mathbf p]` (7 entries). The block mass matrix is

.. math::

    \mathbf M_B = \begin{bmatrix} m\,\mathbf I_3 & \mathbf 0 \\
    \mathbf 0 & 4\,G^\mathsf{T}\bar{\mathbf J}\,G \end{bmatrix},

and the quadratic-velocity (centrifugal/Coriolis) inertia force is

.. math::

    (\mathbf Q_v)_R = \mathbf 0, \qquad
    (\mathbf Q_v)_p = -4\,G^\mathsf{T}\bar{\mathbf J}\,\dot G\,\dot{\mathbf p}
    - 2\,G^\mathsf{T}\bigl(\boldsymbol\omega'\times\bar{\mathbf J}\,\boldsymbol\omega'\bigr),

obtained by premultiplying the body-frame Newton–Euler equation by
:math:`2\,G^\mathsf{T}` and using :math:`\boldsymbol\omega' = 2 G\dot{\mathbf p}`,
:math:`\dot G = G(\dot{\mathbf p})`.

Reference: Shabana, *Computational Dynamics*, 3rd ed., eq. (6.142)–(6.146).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..math import quaternions as _q

if TYPE_CHECKING:
    from ..constraints.base import Constraint
    from .coordinates import SystemCoordinates

__all__ = [
    "AugmentedSystem",
    "assemble_augmented",
    "constraint_jacobian",
    "constraint_residual",
    "constraint_rhs",
    "mass_matrix",
    "quadratic_velocity_forces",
    "solve_accelerations",
]


# ─── Mass matrix and inertia forces ───────────────────────────────────


def mass_matrix(layout: SystemCoordinates) -> NDArray[np.float64]:
    r"""Assemble the block-diagonal generalized mass matrix, shape ``(7n, 7n)``.

    Each body contributes ``diag(m I_3, 4 G^T J̄ G)`` on the diagonal.
    """
    n = layout.n_coords
    M = np.zeros((n, n), dtype=np.float64)
    for body in layout.bodies:
        s = layout.offset(body)
        M[s : s + 3, s : s + 3] = body.mass * np.eye(3)
        G = _q.G(body.quaternion)
        M[s + 3 : s + 7, s + 3 : s + 7] = 4.0 * G.T @ body.inertia @ G
    return M


def quadratic_velocity_forces(layout: SystemCoordinates) -> NDArray[np.float64]:
    r"""Assemble the quadratic-velocity inertia force :math:`\mathbf Q_v`, shape ``(7n,)``.

    The translational part is zero; the rotational part is
    :math:`-4 G^\mathsf{T}\bar{\mathbf J}\dot G\dot{\mathbf p}
    - 2 G^\mathsf{T}(\boldsymbol\omega'\times\bar{\mathbf J}\boldsymbol\omega')`.
    """
    Qv = np.zeros(layout.n_coords, dtype=np.float64)
    for body in layout.bodies:
        s = layout.offset(body)
        p = body.quaternion
        w = body.omega_body
        J = body.inertia
        pdot = _q.omega_body_to_pdot(p, w)
        G = _q.G(p)
        Gdot = _q.G(pdot)
        Qv[s + 3 : s + 7] = -4.0 * G.T @ J @ Gdot @ pdot - 2.0 * G.T @ np.cross(w, J @ w)
    return Qv


# ─── Constraint assembly ──────────────────────────────────────────────


def n_constraint_eqs(constraints: list[Constraint], layout: SystemCoordinates) -> int:
    """Total constraint rows: joint equations plus one normalization per body."""
    return sum(c.n_eq for c in constraints) + layout.n_bodies


def constraint_jacobian(
    constraints: list[Constraint], layout: SystemCoordinates
) -> NDArray[np.float64]:
    r"""Assemble the constraint Jacobian :math:`\boldsymbol\Phi_q`, shape ``(m, 7n)``.

    Rows are ordered: all joint-constraint rows (in ``constraints`` order),
    then one Euler-parameter normalization row per body (in layout order),
    whose ``p``-block is :math:`2\,\mathbf p^\mathsf{T}`.
    """
    m = n_constraint_eqs(constraints, layout)
    Phi_q = np.zeros((m, layout.n_coords), dtype=np.float64)

    row = 0
    for c in constraints:
        for body, J_R, J_p in c.phi_q():
            if not _in_layout(body, layout):
                continue
            s = layout.offset(body)
            Phi_q[row : row + c.n_eq, s : s + 3] += J_R
            Phi_q[row : row + c.n_eq, s + 3 : s + 7] += J_p
        row += c.n_eq

    for body in layout.bodies:
        s = layout.offset(body)
        Phi_q[row, s + 3 : s + 7] = 2.0 * body.quaternion
        row += 1

    return Phi_q


def constraint_residual(
    constraints: list[Constraint], layout: SystemCoordinates
) -> NDArray[np.float64]:
    r"""Assemble the constraint residual :math:`\boldsymbol\Phi`, shape ``(m,)``.

    Joint residuals followed by the normalization residuals
    :math:`\mathbf p^\mathsf{T}\mathbf p - 1` (zero for unit quaternions).
    """
    m = n_constraint_eqs(constraints, layout)
    phi = np.empty(m, dtype=np.float64)
    row = 0
    for c in constraints:
        phi[row : row + c.n_eq] = c.phi()
        row += c.n_eq
    for body in layout.bodies:
        p = body.quaternion
        phi[row] = float(p @ p) - 1.0
        row += 1
    return phi


def constraint_rhs(constraints: list[Constraint], layout: SystemCoordinates) -> NDArray[np.float64]:
    r"""Assemble the acceleration-level RHS :math:`\boldsymbol\gamma`, shape ``(m,)``.

    Joint :math:`\gamma` values followed by the normalization RHS
    :math:`-2\,\dot{\mathbf p}^\mathsf{T}\dot{\mathbf p}`.
    """
    m = n_constraint_eqs(constraints, layout)
    gamma = np.empty(m, dtype=np.float64)
    row = 0
    for c in constraints:
        gamma[row : row + c.n_eq] = c.gamma()
        row += c.n_eq
    for body in layout.bodies:
        pdot = _q.omega_body_to_pdot(body.quaternion, body.omega_body)
        gamma[row] = -2.0 * float(pdot @ pdot)
        row += 1
    return gamma


# ─── Augmented system ─────────────────────────────────────────────────


class AugmentedSystem(NamedTuple):
    """The assembled saddle-point system and its block sizes.

    Attributes
    ----------
    A
        Coefficient matrix ``[[M, Φ_qᵀ], [Φ_q, 0]]``, shape ``(7n+m, 7n+m)``.
    b
        Right-hand side ``[Q_e + Q_v; γ]``, shape ``(7n+m,)``.
    n_coords
        Number of generalized coordinates ``7n``.
    n_eqs
        Number of constraint equations ``m``.
    """

    A: NDArray[np.float64]
    b: NDArray[np.float64]
    n_coords: int
    n_eqs: int


def assemble_augmented(
    layout: SystemCoordinates,
    constraints: list[Constraint],
    applied_forces: ArrayLike | None = None,
) -> AugmentedSystem:
    r"""Assemble the full augmented DAE system at the current state.

    Parameters
    ----------
    layout
        Coordinate layout mapping bodies to global indices.
    constraints
        Joint constraints (normalization is appended automatically).
    applied_forces
        Optional generalized applied force :math:`\mathbf Q_e`, shape
        ``(7n,)``. Defaults to zero.

    Returns
    -------
    AugmentedSystem
        The coefficient matrix, right-hand side, and block sizes.
    """
    n = layout.n_coords
    m = n_constraint_eqs(constraints, layout)

    M = mass_matrix(layout)
    Phi_q = constraint_jacobian(constraints, layout)
    Qv = quadratic_velocity_forces(layout)
    gamma = constraint_rhs(constraints, layout)

    if applied_forces is None:
        Qe = np.zeros(n, dtype=np.float64)
    else:
        Qe = np.asarray(applied_forces, dtype=np.float64).reshape(n)

    A = np.zeros((n + m, n + m), dtype=np.float64)
    A[:n, :n] = M
    A[:n, n:] = Phi_q.T
    A[n:, :n] = Phi_q

    b = np.empty(n + m, dtype=np.float64)
    b[:n] = Qe + Qv
    b[n:] = gamma

    return AugmentedSystem(A=A, b=b, n_coords=n, n_eqs=m)


def solve_accelerations(
    layout: SystemCoordinates,
    constraints: list[Constraint],
    applied_forces: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Solve the augmented system for accelerations and multipliers.

    Returns
    -------
    qddot : ndarray, shape ``(7n,)``
        Generalized accelerations :math:`\ddot{\mathbf q}`.
    lam : ndarray, shape ``(m,)``
        Lagrange multipliers :math:`\boldsymbol\lambda`.
    """
    sys = assemble_augmented(layout, constraints, applied_forces)
    sol = np.linalg.solve(sys.A, sys.b)
    return sol[: sys.n_coords], sol[sys.n_coords :]


# ─── Helpers ──────────────────────────────────────────────────────────


def _in_layout(body: object, layout: SystemCoordinates) -> bool:
    """Return ``True`` if ``body`` has a slot in ``layout``."""
    try:
        layout.offset(body)  # type: ignore[arg-type]
    except KeyError:
        return False
    return True
