"""Conversion of the generalized-alpha trajectory to the common result format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..dynamics import quasi as _quasi
from .result import SimulationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..constraints.base import Constraint
    from ..dynamics.coordinates import SystemCoordinates
    from ..forces.base import Force
    from .dynamics import DynamicsResult

__all__ = ["simulation_from_dynamics"]


def simulation_from_dynamics(
    layout: SystemCoordinates,
    constraints: Sequence[Constraint],
    forces: Sequence[Force],
    result: DynamicsResult,
) -> SimulationResult:
    r"""Express a :class:`~modyn.solver.dynamics.DynamicsResult` as a :class:`SimulationResult`.

    The quasi-velocities follow from :math:`\boldsymbol\omega' = 2G\dot{\mathbf p}`;
    quasi-accelerations and multipliers on all rows are recomputed at every
    stored state with :func:`~modyn.dynamics.quasi.solve_accelerations_quasi`
    (minimum-norm multipliers), so that both solvers are post-processed in
    the same way.
    """
    cons = list(constraints)
    nt = result.times.size
    n_vel = _quasi.n_velocities(layout)
    m = sum(c.n_eq for c in cons)

    V = np.empty((nt, n_vel))
    vdot = np.empty((nt, n_vel))
    lam = np.empty((nt, m))
    for k in range(nt):
        t = float(result.times[k])
        layout.scatter_q(result.q[k], normalize=False)
        V[k] = _quasi.velocity_from_qdot(layout, result.qdot[k])
        _quasi.scatter_velocity(layout, V[k])
        for c in cons:
            c.set_time(t)
        Qe = sum((f.generalized_force(layout, t) for f in forces), np.zeros(layout.n_coords))
        vdot[k], lam[k] = _quasi.solve_accelerations_quasi(layout, cons, Qe)

    out = SimulationResult(
        times=result.times.copy(),
        q=result.q.copy(),
        v=V,
        vdot=vdot,
        lam=lam,
        method="generalized_alpha",
        success=result.success,
    )
    out.info.update(
        newton_iterations=result.newton_iters,
        quaternion_norm_error=out.quaternion_norm_error(),
    )
    return out
