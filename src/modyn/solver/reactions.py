r"""Joint reactions from the constraint multipliers.

The constraint force of joint :math:`k` on the quasi-velocities of a body is
:math:`-\mathbf D_{k,B}^\mathsf{T}\boldsymbol\lambda_k = [\mathbf F,\ \boldsymbol\tau']`:
the force on the centroid (global frame) and the torque about the centroid
(body frame). This module turns it into the force and torque exerted by a
joint on one of its two bodies, referred to the origin of that side's marker,
in the global frame or in the marker frame.

For a joint with the ground on one side the wrench on the ground follows from
equilibrium of the (massless) joint: it is the opposite of the wrench on the
moving body, with the torque transported to the ground marker origin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..math import quaternions as _q
from ..model.ground import Ground

if TYPE_CHECKING:
    from ..constraints.base import Constraint
    from ..symbolic import SymbolicSystem
    from .result import SimulationResult

__all__ = ["joint_reaction"]


def _row_offset(system: SymbolicSystem, joint: Constraint) -> int:
    offset = 0
    for c in system.constraints:
        if c is joint:
            return offset
        offset += c.n_eq
    raise ValueError("joint is not part of the symbolic system")


def joint_reaction(
    system: SymbolicSystem,
    result: SimulationResult,
    joint: Constraint,
    side: str = "i",
    frame: str = "global",
) -> NDArray[np.float64]:
    r"""Force and torque exerted by ``joint`` on the body of its marker ``side``.

    Parameters
    ----------
    system
        Symbolic system the result was computed with (provides :math:`\mathbf D`).
    result
        Simulation result (positions, times and multipliers on all rows).
    joint
        A joint with ``marker_i`` and ``marker_j``.
    side
        ``"i"`` or ``"j"``: which marker's body receives the reaction.
    frame
        ``"global"`` or ``"marker"`` (the orientation of the chosen marker).

    Returns
    -------
    ndarray, shape ``(nt, 6)``
        Force (columns 0-2) and torque about the marker origin (columns 3-5).
    """
    if side not in ("i", "j"):
        raise ValueError("side must be 'i' or 'j'")
    if frame not in ("global", "marker"):
        raise ValueError("frame must be 'global' or 'marker'")

    m_i, m_j = joint.marker_i, joint.marker_j
    marker = m_i if side == "i" else m_j
    other = m_j if side == "i" else m_i
    layout = system.layout
    o = _row_offset(system, joint)
    n = joint.n_eq
    nt = result.times.size
    out = np.zeros((nt, 6))

    def wrench_on(mk, k: int):
        """Wrench of the joint on the (moving) body of ``mk`` about the marker origin, global."""
        body = mk.parent
        b = layout.offset(body) // 7
        D = system.jacobian(result.q[k], result.times[k])[o : o + n, 6 * b : 6 * b + 6]
        w = -D.T @ result.lam[k, o : o + n]
        R = result.q[k, 7 * b : 7 * b + 3]
        A = _q.A(result.q[k, 7 * b + 3 : 7 * b + 7])
        F = w[:3]
        arm = A @ mk.position_local
        tau = A @ w[3:] - np.cross(arm, F)
        return F, tau, A @ mk.orientation_local, R + arm

    for k in range(nt):
        if not isinstance(marker.parent, Ground):
            F, tau, R_mk, _ = wrench_on(marker, k)
        else:
            # Equal and opposite wrench on the ground, torque moved to its marker origin.
            F_o, tau_o, _, P_o = wrench_on(other, k)
            P_g = np.asarray(marker.position_local)
            F = -F_o
            tau = -tau_o - np.cross(P_o - P_g, F_o)
            R_mk = marker.orientation_local
        if frame == "marker":
            F, tau = R_mk.T @ F, R_mk.T @ tau
        out[k, :3], out[k, 3:] = F, tau
    return out
