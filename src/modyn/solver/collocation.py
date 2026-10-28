r"""Direct index-3 integration by Radau IIA collocation (CasADi).

This is the solver of the hybrid formulation, the 3D counterpart of the one
used by PMD. The constraints are kept at position level and integrated
together with the equations of motion as the semi-explicit DAE built by
:meth:`modyn.symbolic.system.SymbolicSystem.dae`,

.. math::

    \dot{\mathbf q} = \mathbf T\mathbf v + \mathbf P\boldsymbol\mu,\qquad
    \mathbf M\dot{\mathbf v} = \mathbf h - \mathbf D^\mathsf{T}\boldsymbol\lambda,\qquad
    \mathbf 0 = [\boldsymbol\Phi(\mathbf q, t);\ \mathbf p_k^\mathsf{T}\mathbf p_k - 1],

with one finite element of the 3-stage Radau IIA scheme per reporting step.
The Jacobian :math:`\mathbf D`, the explicit time derivatives and the
acceleration terms come from automatic differentiation. Redundant constraint
rows are detected on the assembled configuration and removed from the DAE;
the multipliers on all rows are recovered a posteriori (minimum norm).
"""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from ..dynamics import quasi as _quasi
from ..dynamics.redundancy import RowSelection, select_independent_rows
from ..symbolic import SymbolicSystem
from .result import SimulationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..constraints.base import Constraint
    from ..dynamics.coordinates import SystemCoordinates
    from ..forces.base import Force

__all__ = ["integrate_radau"]


def integrate_radau(
    layout: SystemCoordinates,
    constraints: Sequence[Constraint],
    forces: Sequence[Force],
    t_span: tuple[float, float],
    dt: float,
    *,
    rows: str | Sequence[int] = "auto",
    project_initial: bool = True,
    rank_tol: float = 1e-9,
    rootfinder_options: dict | None = None,
    progress: bool = False,
) -> SimulationResult:
    r"""Integrate the model with Radau IIA collocation on a uniform grid.

    Parameters
    ----------
    layout
        Coordinate layout; the bodies must hold an *assembled* configuration
        (use :meth:`modyn.solver.model.SpatialMultibodyModel.assemble`).
    constraints, forces
        Constraints and forces (all must have a symbolic formulation).
    t_span
        ``(t0, tf)``.
    dt
        Step of the uniform output grid; one collocation element is taken per
        step, as in PMD.
    rows
        ``"auto"`` removes redundant constraint rows (rank-revealing QR on
        the assembled configuration), ``"all"`` keeps every row, or an
        explicit list of row indices to keep.
    project_initial
        Correct the initial velocities so that
        :math:`\mathbf D\mathbf v + \boldsymbol\Phi_t = \mathbf 0`.
    rank_tol
        Threshold of the redundancy analysis.
    rootfinder_options
        Extra options for the KINSOL root finder of the collocation system
        (default ``abstol=1e-10``).
    progress
        Show a progress bar.

    Returns
    -------
    SimulationResult
        Positions, quasi-velocities, quasi-accelerations and multipliers on
        all rows. ``info`` holds the kept/dropped rows, the maximum
        constraint violation and the wall-clock time.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    if tf <= t0:
        raise ValueError(f"t_span must satisfy t0 < tf, got ({t0}, {tf}).")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}.")
    n_steps = int(round((tf - t0) / dt))
    if n_steps < 1 or abs(n_steps * dt - (tf - t0)) > 1e-9 * max(1.0, abs(tf)):
        raise ValueError("dt must divide the time span (uniform output grid)")
    times = t0 + dt * np.arange(n_steps + 1)

    clock = _time.perf_counter()
    cons = list(constraints)
    for c in cons:
        c.set_time(t0)
    sym = SymbolicSystem(layout, cons, list(forces))

    q = layout.assemble_q()
    v = _quasi.assemble_velocity(layout)

    # ── Consistent initial values ─────────────────────────────────────
    D0 = sym.jacobian(q, t0)
    if sym.n_eq and project_initial:
        nu0 = D0 @ v + sym.phi_t(q, t0)
        v = v - np.linalg.lstsq(D0, nu0, rcond=None)[0]

    if isinstance(rows, str):
        if rows == "auto":
            selection = select_independent_rows(D0, tol=rank_tol)
        elif rows == "all":
            idx = np.arange(sym.n_eq, dtype=np.intp)
            selection = RowSelection(idx, np.zeros(0, dtype=np.intp), sym.n_eq)
        else:
            raise ValueError("rows must be 'auto', 'all' or a list of indices")
    else:
        idx = np.asarray(sorted(int(r) for r in rows), dtype=np.intp)
        selection = RowSelection(idx, np.setdiff1d(np.arange(sym.n_eq), idx), int(idx.size))
    kept = selection.rows

    _, lam0 = sym.solve_accelerations(q, v, t0, rows=kept)
    z = np.concatenate((lam0, np.zeros(layout.n_bodies)))

    # ── Collocation integrator (one Radau IIA element per step) ───────
    opts = {
        "rootfinder": "kinsol",
        "collocation_scheme": "radau",
        "number_of_finite_elements": 1,
        "rootfinder_options": {"abstol": 1e-10, **(rootfinder_options or {})},
    }
    step = ca.integrator("radau_step", "collocation", sym.dae(kept), 0.0, dt, opts)

    Q = np.empty((n_steps + 1, sym.n_coords))
    V = np.empty((n_steps + 1, sym.n_vel))
    Q[0], V[0] = q, v
    x = np.concatenate((q, v, [t0]))

    success = True
    last = n_steps
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=n_steps, desc="Radau IIA", colour="green")
    for k in range(n_steps):
        try:
            sol = step(x0=x, z0=z)
        except RuntimeError:
            success = False
            last = k
            break
        x = np.asarray(sol["xf"]).ravel()
        z = np.asarray(sol["zf"]).ravel()
        x[-1] = times[k + 1]
        Q[k + 1] = x[: sym.n_coords]
        V[k + 1] = x[sym.n_coords : sym.n_coords + sym.n_vel]
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    times, Q, V = times[: last + 1], Q[: last + 1], V[: last + 1]

    # ── Post-processing: accelerations, reactions, constraint drift ───
    nt = times.size
    vdot = np.empty((nt, sym.n_vel))
    lam = np.empty((nt, sym.n_eq))
    viol = np.empty(nt)
    for k in range(nt):
        vdot[k], lam[k] = sym.solve_accelerations(Q[k], V[k], times[k])
        viol[k] = np.max(np.abs(sym.phi(Q[k], times[k]))) if sym.n_eq else 0.0

    layout.scatter_q(Q[-1], normalize=False)
    _quasi.scatter_velocity(layout, V[-1])

    result = SimulationResult(
        times=times, q=Q, v=V, vdot=vdot, lam=lam, method="radau", success=success
    )
    result.info.update(
        kept_rows=kept,
        dropped_rows=selection.dropped,
        rank=selection.rank,
        max_constraint_violation=float(viol.max()),
        constraint_violation=viol,
        quaternion_norm_error=result.quaternion_norm_error(),
        wall_time=_time.perf_counter() - clock,
    )
    return result
