r"""Translational Spring-Damper-Actuator (TSDA) element.

A TSDA connects two markers and applies equal-and-opposite forces along the
line joining them.  The scalar force magnitude is

.. math::

    F = k\,(l - l_0) + c\,\dot l + F_\text{act}(t),

where :math:`l` is the current distance, :math:`l_0` the natural length,
:math:`k` the stiffness, :math:`c` the damping coefficient, and
:math:`F_\text{act}` an optional actuator force.

The corresponding generalized forces follow from virtual work: a force
:math:`F\,\mathbf e` (with :math:`\mathbf e` the unit direction vector from
marker :math:`i` to marker :math:`j`) is applied at the origin of marker
:math:`j`, and :math:`-F\,\mathbf e` at the origin of marker :math:`i`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..core.utils import ensure_non_negative
from ..math import quaternions as _q
from .base import Force

if TYPE_CHECKING:
    from ..dynamics.coordinates import SystemCoordinates
    from ..model.marker import Marker

__all__ = ["TSDA"]


class TSDA(Force):
    r"""Translational Spring-Damper-Actuator between two markers.

    Parameters
    ----------
    marker_i, marker_j
        The two attachment markers. Either parent may be :class:`~modyn.model.ground.Ground`.
    k
        Linear stiffness [N/m]. Must be non-negative.
    c
        Linear damping coefficient [N s/m]. Must be non-negative.
        Default ``0.0``.
    natural_length
        Unstretched length :math:`l_0` [m].  When ``None`` (default) the
        natural length is set to the current distance between the markers on
        the first call to :meth:`generalized_force` (lazy initialisation).
    actuator
        Actuator force [N].  Either a constant ``float`` or a callable
        ``f(t) -> float``.  Default ``0.0`` (no actuation).
    """

    def __init__(
        self,
        marker_i: Marker,
        marker_j: Marker,
        *,
        k: float,
        c: float = 0.0,
        natural_length: float | None = None,
        actuator: float | Callable[[float], float] = 0.0,
    ) -> None:
        self._mi = marker_i
        self._mj = marker_j
        self._k = float(ensure_non_negative(k, "k"))
        self._c = float(ensure_non_negative(c, "c"))
        self._l0: float | None = None if natural_length is None else float(natural_length)
        if callable(actuator):
            self._F_act: Callable[[float], float] = actuator
        else:
            _fa = float(actuator)
            self._F_act = lambda t: _fa

    @property
    def marker_i(self) -> Marker:
        """First attachment marker."""
        return self._mi

    @property
    def marker_j(self) -> Marker:
        """Second attachment marker."""
        return self._mj

    @property
    def natural_length(self) -> float | None:
        r"""Unstretched length :math:`l_0` [m] (``None`` until first evaluation)."""
        return self._l0

    def generalized_force(
        self,
        layout: SystemCoordinates,
        t: float,
    ) -> NDArray[np.float64]:
        r"""Compute the TSDA generalized-force contribution.

        For each marker :math:`P` (at body-frame position :math:`\bar{\mathbf s}`):

        .. math::

            (\mathbf Q_e)_R = \pm F\,\mathbf e, \qquad
            (\mathbf Q_e)_p = 2\,G^\mathsf{T}
                (\bar{\mathbf s} \times \mathbf A^\mathsf{T}(\pm F\,\mathbf e)).
        """
        bi, bj = self._mi.parent, self._mj.parent
        s_i = self._mi.position_local
        s_j = self._mj.position_local

        r_i = bi.point_global(s_i)
        r_j = bj.point_global(s_j)
        d = r_j - r_i
        l = float(np.linalg.norm(d))

        if l < 1e-14:
            return np.zeros(layout.n_coords, dtype=np.float64)

        e = d / l  # unit vector from i to j

        # Lazy natural length
        if self._l0 is None:
            self._l0 = l

        # Length rate: l̇ = e · (v_j^P - v_i^P)
        v_i = bi.velocity_of_point(s_i)
        v_j = bj.velocity_of_point(s_j)
        l_dot = float(e @ (v_j - v_i))

        F = self._k * (l - self._l0) + self._c * l_dot + self._F_act(t)
        Qe = np.zeros(layout.n_coords, dtype=np.float64)

        for body, s_body, sign in ((bi, s_i, -1.0), (bj, s_j, 1.0)):
            try:
                s = layout.offset(body)  # type: ignore[arg-type]
            except KeyError:
                continue  # Ground has no DOFs in the layout
            F_vec = sign * F * e
            p = body.quaternion
            G = _q.G(p)
            A = _q.A(p)
            tau_body = np.cross(s_body, A.T @ F_vec)
            Qe[s : s + 3] += F_vec
            Qe[s + 3 : s + 7] += 2.0 * G.T @ tau_body

        return Qe

    def wrench_sym(self, state, t):
        """Equal and opposite forces along the marker line, with their body-frame torques."""
        import casadi as ca

        bi, bj = self._mi.parent, self._mj.parent
        s_i, s_j = self._mi.position_local, self._mj.position_local
        if self._l0 is None:
            self._l0 = float(np.linalg.norm(bj.point_global(s_j) - bi.point_global(s_i)))

        d = state.point(bj, s_j) - state.point(bi, s_i)
        length = ca.sqrt(ca.mtimes(d.T, d))
        e = d / length
        l_dot = ca.mtimes(e.T, state.point_velocity(bj, s_j) - state.point_velocity(bi, s_i))
        F = self._k * (length - self._l0) + self._c * l_dot + self._F_act(t)

        out = []
        for body, s_body, sign in ((bi, s_i, -1.0), (bj, s_j, 1.0)):
            if state._is_ground(body):
                continue
            F_vec = sign * F * e
            tau = ca.cross(ca.DM(s_body), ca.mtimes(state.A(body).T, F_vec))
            out.append((body, F_vec, tau))
        return out
