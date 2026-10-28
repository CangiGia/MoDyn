r"""Kinematic analysis — position assembly and velocity consistency.

These drivers make the state currently stored on the bodies *consistent*
with the constraints, which is the standard preparation step before a
dynamic simulation:

- :func:`assemble_position` solves :math:`\boldsymbol\Phi(\mathbf q)=\mathbf 0`
  by Newton iteration, projecting an approximate initial guess onto the
  constraint manifold. The (generally rectangular, ``m \le 7n``) Jacobian
  is handled with the minimum-norm Gauss–Newton step
  :math:`\Delta\mathbf q = -\boldsymbol\Phi_q^{+}\,\boldsymbol\Phi`, so the
  routine works both for kinematically determined systems (``m = 7n``) and
  for systems with remaining degrees of freedom.

- :func:`solve_velocity` enforces the velocity-level constraint
  :math:`\boldsymbol\Phi_q\,\dot{\mathbf q} = \boldsymbol\nu` (with
  :math:`\boldsymbol\nu = \mathbf 0` for the scleronomic constraints
  available today) by correcting the stored velocities with the smallest
  change, i.e. projecting them onto the constraint tangent space.

The per-body Euler-parameter normalization is included automatically by the
assembly helpers in :mod:`modyn.dynamics.eom`.

Reference: Shabana, *Computational Dynamics*, 3rd ed., §3.5 and §6.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..dynamics.eom import constraint_jacobian, constraint_residual

if TYPE_CHECKING:
    from ..constraints.base import Constraint
    from ..dynamics.coordinates import SystemCoordinates

__all__ = ["AssemblyResult", "assemble_position", "solve_velocity"]


class AssemblyResult(NamedTuple):
    """Outcome of a position-assembly Newton iteration.

    Attributes
    ----------
    converged
        ``True`` if the residual norm dropped to ``tol`` within ``max_iter``.
    iterations
        Number of Newton steps actually performed.
    residual_norm
        Final Euclidean norm of the constraint residual.
    """

    converged: bool
    iterations: int
    residual_norm: float


def assemble_position(
    layout: SystemCoordinates,
    constraints: list[Constraint],
    *,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> AssemblyResult:
    r"""Project the current configuration onto the constraint manifold.

    Performs Newton iteration on :math:`\boldsymbol\Phi(\mathbf q)=\mathbf 0`,
    mutating the bodies in ``layout`` in place. Each step takes the
    minimum-norm correction
    :math:`\Delta\mathbf q = -\boldsymbol\Phi_q^{+}\,\boldsymbol\Phi`
    (via :func:`numpy.linalg.lstsq`) and renormalizes every quaternion when
    the new coordinates are scattered back onto the bodies.

    Parameters
    ----------
    layout
        Coordinate layout; its bodies hold the state that is updated.
    constraints
        Joint constraints (normalization is appended automatically).
    tol
        Convergence threshold on :math:`\lVert\boldsymbol\Phi\rVert_2`.
    max_iter
        Maximum number of Newton steps.

    Returns
    -------
    AssemblyResult
        Convergence flag, iteration count, and final residual norm.
    """
    if tol <= 0.0:
        raise ValueError("tol must be strictly positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    residual_norm = float(np.linalg.norm(constraint_residual(constraints, layout)))
    for iteration in range(1, max_iter + 1):
        if residual_norm <= tol:
            return AssemblyResult(True, iteration - 1, residual_norm)

        phi = constraint_residual(constraints, layout)
        Phi_q = constraint_jacobian(constraints, layout)
        delta_q, *_ = np.linalg.lstsq(Phi_q, -phi, rcond=None)
        layout.scatter_q(layout.assemble_q() + delta_q)
        residual_norm = float(np.linalg.norm(constraint_residual(constraints, layout)))

    return AssemblyResult(residual_norm <= tol, max_iter, residual_norm)


def solve_velocity(
    layout: SystemCoordinates,
    constraints: list[Constraint],
    rhs: ArrayLike | None = None,
) -> NDArray[np.float64]:
    r"""Make the stored velocities satisfy the velocity-level constraint.

    Corrects the current :math:`\dot{\mathbf q}` with the smallest change
    that enforces :math:`\boldsymbol\Phi_q\,\dot{\mathbf q} = \boldsymbol\nu`:

    .. math::

        \dot{\mathbf q} \leftarrow \dot{\mathbf q}_0
        + \boldsymbol\Phi_q^{+}\,(\boldsymbol\nu - \boldsymbol\Phi_q\,\dot{\mathbf q}_0).

    With ``rhs = None`` (the scleronomic case) this projects the velocities
    onto the constraint tangent space, in particular forcing each quaternion
    rate onto the unit-sphere tangent (:math:`\mathbf p^\mathsf{T}\dot{\mathbf p}=0`).
    The corrected velocities are scattered back onto the bodies.

    Parameters
    ----------
    layout
        Coordinate layout; its bodies hold the velocities that are updated.
    constraints
        Joint constraints (normalization is appended automatically).
    rhs
        Optional velocity-constraint right-hand side :math:`\boldsymbol\nu`,
        shape ``(m,)``. Defaults to zero.

    Returns
    -------
    ndarray, shape ``(7n,)``
        The corrected global velocity vector.
    """
    Phi_q = constraint_jacobian(constraints, layout)
    m = Phi_q.shape[0]
    nu = (
        np.zeros(m, dtype=np.float64)
        if rhs is None
        else np.asarray(rhs, dtype=np.float64).reshape(m)
    )

    qdot0 = layout.assemble_qdot()
    correction, *_ = np.linalg.lstsq(Phi_q, nu - Phi_q @ qdot0, rcond=None)
    qdot = qdot0 + correction
    layout.scatter_qdot(qdot)
    return qdot
