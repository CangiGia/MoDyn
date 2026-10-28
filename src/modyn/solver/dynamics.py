r"""Dynamic (time-domain) analysis — direct index-3 DAE integration.

The public entry point is :func:`integrate_dynamics`, which wraps
:class:`~modyn.solver.integrators.GeneralizedAlpha` in a fixed-step loop
that:

1. Assembles consistent initial conditions via
   :func:`~modyn.solver.kinematics.assemble_position`,
   :func:`~modyn.solver.kinematics.solve_velocity`, and
   :func:`~modyn.dynamics.eom.solve_accelerations`.
2. Evaluates the equation-of-motion quantities at each trial state and feeds
   them to the Newton iteration inside the integrator.
3. Stores the full trajectory (coordinates, velocities, accelerations,
   multipliers) in a :class:`DynamicsResult` NamedTuple.

No Baumgarte stabilization or post-step projection is applied. The
position-level constraints :math:`\boldsymbol\Phi(\mathbf q) = \mathbf 0`
are enforced at every step by the Newton solver inside the integrator, so
the constraint drift is bounded at machine precision.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..dynamics.eom import (
    constraint_jacobian,
    constraint_residual,
    mass_matrix,
    quadratic_velocity_forces,
    solve_accelerations,
)
from .integrators import GeneralizedAlpha
from .kinematics import assemble_position, solve_velocity

if TYPE_CHECKING:
    from ..constraints.base import Constraint
    from ..dynamics.coordinates import SystemCoordinates

__all__ = ["DynamicsResult", "integrate_dynamics"]

# A force specification is either a constant (7n,) array or a callable
# ``f(t, q, qdot) -> (7n,)``.
ForceSpec = (
    ArrayLike | Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
)


class DynamicsResult(NamedTuple):
    r"""Trajectory returned by :func:`integrate_dynamics`.

    Attributes
    ----------
    times
        Time instants, shape ``(nt,)``, where ``nt`` is the number of
        recorded steps (including :math:`t_0`).
    q
        Generalized coordinates at each instant, shape ``(nt, 7n)``.
    qdot
        Generalized velocities at each instant, shape ``(nt, 7n)``.
    qddot
        Generalized accelerations at each instant, shape ``(nt, 7n)``.
    lam
        Lagrange multipliers at each instant, shape ``(nt, m)``.
    newton_iters
        Number of Newton iterations performed at each step, shape ``(nt,)``.
        The first entry (for :math:`t_0`) is 0.
    residual_norm
        Newton residual norm at each step, shape ``(nt,)``.
        The first entry is the initial residual before any iteration.
    success
        ``True`` if every step converged within the requested tolerance.
    """

    times: NDArray[np.float64]
    q: NDArray[np.float64]
    qdot: NDArray[np.float64]
    qddot: NDArray[np.float64]
    lam: NDArray[np.float64]
    newton_iters: NDArray[np.intp]
    residual_norm: NDArray[np.float64]
    success: bool


def integrate_dynamics(
    layout: SystemCoordinates,
    constraints: list[Constraint],
    t_span: tuple[float, float],
    dt: float,
    *,
    applied_forces: ForceSpec | None = None,
    rho_inf: float = 0.8,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    project_initial: bool = True,
    assembly_tol: float = 1e-10,
    assembly_max_iter: int = 50,
) -> DynamicsResult:
    r"""Integrate the augmented index-3 multibody DAE.

    Parameters
    ----------
    layout
        :class:`~modyn.dynamics.coordinates.SystemCoordinates` instance
        that maps bodies to global index slots.
    constraints
        List of joint constraints. The per-body Euler-parameter
        normalization :math:`\mathbf p^\mathsf{T}\mathbf p - 1 = 0` is
        appended automatically by the EOM assembly functions.
    t_span
        ``(t0, tf)`` — start and end times.
    dt
        Fixed step size.
    applied_forces
        Generalized applied forces :math:`\mathbf Q_e`. May be:

        - ``None`` — zero (free or constraint-only simulation).
        - An array of shape ``(7n,)`` — constant force vector.
        - A callable ``f(t, q, qdot) -> ndarray`` of shape ``(7n,)`` —
          time- and state-dependent forces.
    rho_inf
        Spectral radius at infinite frequency for
        :class:`~modyn.solver.integrators.GeneralizedAlpha`,
        :math:`\rho_\infty \in [0, 1]`. Default ``0.8``.
    newton_tol
        Convergence tolerance for the per-step Newton iteration.
    newton_max_iter
        Maximum number of Newton iterations per step.
    project_initial
        When ``True`` (default), call
        :func:`~modyn.solver.kinematics.assemble_position` and
        :func:`~modyn.solver.kinematics.solve_velocity` before computing
        consistent initial accelerations, so that the initial state satisfies
        the constraints exactly.
    assembly_tol, assembly_max_iter
        Tolerance and iteration limit forwarded to
        :func:`~modyn.solver.kinematics.assemble_position` when
        ``project_initial=True``.

    Returns
    -------
    DynamicsResult
        The full trajectory and per-step Newton diagnostics.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    if tf <= t0:
        raise ValueError(f"t_span must satisfy t0 < tf, got ({t0}, {tf}).")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}.")
    if not (0.0 <= rho_inf <= 1.0):
        raise ValueError(f"rho_inf must lie in [0, 1], got {rho_inf!r}.")

    n = layout.n_coords

    # ── Normalise the force specification ─────────────────────────────
    if applied_forces is None:
        _Qe_zeros = np.zeros(n, dtype=np.float64)

        def _force_fn(
            t: float, q: NDArray[np.float64], qdot: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return _Qe_zeros

    elif callable(applied_forces):
        _user_fn = applied_forces

        def _force_fn(
            t: float, q: NDArray[np.float64], qdot: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return np.asarray(_user_fn(t, q, qdot), dtype=np.float64).reshape(n)

    else:
        _Qe_const = np.asarray(applied_forces, dtype=np.float64).reshape(n)

        def _force_fn(
            t: float, q: NDArray[np.float64], qdot: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return _Qe_const

    # ── Consistent initial conditions ─────────────────────────────────
    if project_initial:
        assemble_position(
            layout,
            constraints,
            tol=assembly_tol,
            max_iter=assembly_max_iter,
        )
        solve_velocity(layout, constraints)

    q0 = layout.assemble_q()
    qdot0 = layout.assemble_qdot()
    Qe0 = _force_fn(t0, q0, qdot0)
    qddot0, lam0 = solve_accelerations(layout, constraints, Qe0)
    a0 = qddot0.copy()

    # ── Build evaluate callback ────────────────────────────────────────
    def _evaluate(
        t: float, q: NDArray[np.float64], qdot: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        # Scatter the trial state without normalising the quaternions —
        # the normalization constraint pᵀp-1=0 is enforced explicitly.
        layout.scatter_q(q, normalize=False)
        layout.scatter_qdot(qdot)
        # Propagate time to rheonomic drivers (no-op for scleronomic joints).
        for c in constraints:
            c.set_time(t)
        M = mass_matrix(layout)
        Phi_q = constraint_jacobian(constraints, layout)
        Qv = quadratic_velocity_forces(layout)
        Qe = _force_fn(t, q, qdot)
        phi = constraint_residual(constraints, layout)
        return M, Phi_q, Qe + Qv, phi

    # ── Time-step loop ─────────────────────────────────────────────────
    integrator = GeneralizedAlpha(rho_inf=rho_inf)

    times_list: list[float] = [t0]
    q_list: list[NDArray[np.float64]] = [q0.copy()]
    qdot_list: list[NDArray[np.float64]] = [qdot0.copy()]
    qddot_list: list[NDArray[np.float64]] = [qddot0.copy()]
    lam_list: list[NDArray[np.float64]] = [lam0.copy()]
    iters_list: list[int] = [0]
    resid_list: list[float] = [0.0]

    q_n = q0.copy()
    qdot_n = qdot0.copy()
    qddot_n = qddot0.copy()
    a_n = a0.copy()
    lam_n = lam0.copy()
    t_n = t0
    success = True

    while t_n + dt <= tf + 1e-12 * abs(tf):
        sr = integrator.step(
            _evaluate,
            t_n,
            dt,
            q_n,
            qdot_n,
            qddot_n,
            a_n,
            lam_n,
            tol=newton_tol,
            max_iter=newton_max_iter,
        )
        if not sr.converged:
            success = False

        q_n = sr.q
        qdot_n = sr.qdot
        qddot_n = sr.qddot
        a_n = sr.a
        lam_n = sr.lam
        t_n = t_n + dt

        times_list.append(t_n)
        q_list.append(q_n.copy())
        qdot_list.append(qdot_n.copy())
        qddot_list.append(qddot_n.copy())
        lam_list.append(lam_n.copy())
        iters_list.append(sr.iterations)
        resid_list.append(sr.residual_norm)

    # Scatter the final state so the bodies reflect the last step
    layout.scatter_q(q_n, normalize=False)
    layout.scatter_qdot(qdot_n)

    return DynamicsResult(
        times=np.array(times_list, dtype=np.float64),
        q=np.array(q_list, dtype=np.float64),
        qdot=np.array(qdot_list, dtype=np.float64),
        qddot=np.array(qddot_list, dtype=np.float64),
        lam=np.array(lam_list, dtype=np.float64),
        newton_iters=np.array(iters_list, dtype=np.intp),
        residual_norm=np.array(resid_list, dtype=np.float64),
        success=success,
    )
