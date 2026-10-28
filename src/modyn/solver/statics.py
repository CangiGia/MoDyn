r"""Static-equilibrium analysis.

Finds a configuration :math:`\mathbf q` and a set of Lagrange multipliers
:math:`\boldsymbol\lambda` satisfying

.. math::

    \mathbf Q_e(\mathbf q) + \boldsymbol\Phi_q(\mathbf q)^\mathsf{T}
        \boldsymbol\lambda = \mathbf 0, \qquad
    \boldsymbol\Phi(\mathbf q) = \mathbf 0,

i.e. the applied forces are in equilibrium with the constraint reactions and
the position-level constraints are satisfied exactly.

**Method** — regularised Newton iteration.  The iteration matrix is the
perturbed saddle-point system

.. math::

    \begin{bmatrix}
        \varepsilon\,\mathbf I & \boldsymbol\Phi_q^\mathsf{T} \\
        \boldsymbol\Phi_q & \mathbf 0
    \end{bmatrix}
    \begin{bmatrix} \Delta\mathbf q \\ \Delta\boldsymbol\lambda \end{bmatrix}
    = -\begin{bmatrix}
        \mathbf Q_e + \boldsymbol\Phi_q^\mathsf{T}\boldsymbol\lambda \\
        \boldsymbol\Phi
      \end{bmatrix}

where the regularisation :math:`\varepsilon` replaces the unknown tangent
stiffness :math:`\partial\mathbf Q_e/\partial\mathbf q`.  For constant
applied forces (e.g. gravity) the omitted term is exactly zero, so the
iteration is consistent.  The per-body Euler-parameter normalisation
:math:`\mathbf p^\mathsf{T}\mathbf p - 1 = 0` is included automatically
through the EOM helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..dynamics.eom import (
    constraint_jacobian,
    constraint_residual,
    n_constraint_eqs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..constraints.base import Constraint
    from ..dynamics.coordinates import SystemCoordinates

__all__ = ["StaticsResult", "find_equilibrium"]


class StaticsResult(NamedTuple):
    r"""Outcome of a static-equilibrium Newton iteration.

    Attributes
    ----------
    converged
        ``True`` when both residuals dropped below ``tol``.
    iterations
        Number of Newton steps performed.
    equilibrium_residual
        Final norm :math:`\lVert\mathbf Q_e + \boldsymbol\Phi_q^\mathsf{T}
        \boldsymbol\lambda\rVert` (force-balance residual).
    constraint_residual
        Final norm :math:`\lVert\boldsymbol\Phi\rVert` (position residual).
    lam
        Lagrange multipliers at the final configuration, shape ``(m,)``.
    """

    converged: bool
    iterations: int
    equilibrium_residual: float
    constraint_residual: float
    lam: NDArray[np.float64]


def find_equilibrium(
    layout: SystemCoordinates,
    constraints: list[Constraint],
    applied_forces: ArrayLike | Callable | None = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
    reg: float = 1e-6,
) -> StaticsResult:
    r"""Find a static-equilibrium configuration.

    The bodies in ``layout`` are updated in-place; their state after the
    call corresponds to the final (converged or last) iterate.

    Parameters
    ----------
    layout
        Coordinate layout.
    constraints
        Joint constraints; normalisation is appended automatically.
    applied_forces
        Generalised applied forces :math:`\mathbf Q_e`.  May be:

        - ``None`` — zero (constraint-only equilibrium; pure kinematic
          assembly).
        - An array of shape ``(7n,)`` — constant force vector.
        - A callable ``f(t, q, qdot) -> ndarray(7n,)`` — same signature as
          accepted by :func:`~modyn.solver.dynamics.integrate_dynamics`.
          Evaluated at ``t=0``, ``qdot=0`` each iteration.
    tol
        Convergence tolerance on both residual norms.
    max_iter
        Maximum number of Newton iterations.
    reg
        Regularisation :math:`\varepsilon` added to the diagonal of the
        :math:`\mathbf q`-block of the iteration matrix.  Replaces the
        unknown tangent stiffness.  Use a small value (default ``1e-6``);
        larger values slow convergence but improve conditioning.

    Returns
    -------
    StaticsResult
    """
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol!r}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}.")
    if reg < 0.0:
        raise ValueError(f"reg must be non-negative, got {reg!r}.")

    n = layout.n_coords
    m = n_constraint_eqs(constraints, layout)

    # ── Normalise the force specification ─────────────────────────────
    _qdot_zero = np.zeros(n, dtype=np.float64)
    if applied_forces is None:

        def _Qe(q: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.zeros(n, dtype=np.float64)
    elif callable(applied_forces):
        _user = applied_forces

        def _Qe(q: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.asarray(_user(0.0, q, _qdot_zero), dtype=np.float64).reshape(n)
    else:
        _const = np.asarray(applied_forces, dtype=np.float64).reshape(n)

        def _Qe(q: NDArray[np.float64]) -> NDArray[np.float64]:
            return _const

    # ── Initial state ─────────────────────────────────────────────────
    q = layout.assemble_q()
    lam = np.zeros(m, dtype=np.float64)

    eq_res = np.inf
    con_res = np.inf
    converged = False
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        layout.scatter_q(q, normalize=False)

        Phi_q = constraint_jacobian(constraints, layout)
        phi = constraint_residual(constraints, layout)
        Qe = _Qe(q)

        r1 = Qe + Phi_q.T @ lam
        eq_res = float(np.linalg.norm(r1))
        con_res = float(np.linalg.norm(phi))
        if eq_res < tol and con_res < tol:
            converged = True
            break

        # Regularised saddle system
        K_reg = reg * np.eye(n, dtype=np.float64)
        St = np.zeros((n + m, n + m), dtype=np.float64)
        St[:n, :n] = K_reg
        St[:n, n:] = Phi_q.T
        St[n:, :n] = Phi_q

        rhs = np.empty(n + m, dtype=np.float64)
        rhs[:n] = -r1
        rhs[n:] = -phi

        delta = np.linalg.lstsq(St, rhs, rcond=None)[0]
        q = q + delta[:n]
        lam = lam + delta[n:]

    # Scatter the final state
    layout.scatter_q(q, normalize=False)

    return StaticsResult(
        converged=converged,
        iterations=iterations,
        equilibrium_residual=eq_res,
        constraint_residual=con_res,
        lam=lam,
    )
