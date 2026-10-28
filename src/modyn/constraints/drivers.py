r"""Rheonomic (time-dependent) drivers.

A driver adds one constraint equation to the system prescribing a scalar
joint coordinate as a function of time.  Drivers are used in combination
with the corresponding scleronomic joint:

- :class:`RevoluteDriver` — prescribes the relative rotation angle between
  two markers (use alongside :class:`~modyn.constraints.RevoluteJoint`).
- :class:`PrismaticDriver` — prescribes the relative displacement along the
  joint axis (use alongside :class:`~modyn.constraints.PrismaticJoint`).

Mathematical formulation
------------------------
For a driver :math:`\Phi_d(\mathbf q, t) = h(\mathbf q) - c(t) = 0`:

- The position residual is :math:`h(\mathbf q) - c(t)`.
- The Jacobian :math:`\partial\Phi_d/\partial\mathbf q` is identical to
  :math:`\partial h/\partial\mathbf q` (time appears only additively).
- The acceleration-level right-hand side is
  :math:`\gamma_d = \gamma_h + \ddot{c}(t)`,
  where :math:`\gamma_h` is the :math:`\ddot{\mathbf q}`-free part of
  :math:`\ddot{h}` for the base constraint :math:`h(\mathbf q) = 0`.

Time injection
--------------
Before calling :meth:`phi`, :meth:`phi_q`, or :meth:`gamma`, the
simulation loop must call :meth:`set_time` to propagate the current
simulation time into the driver.  The
:func:`~modyn.solver.dynamics.integrate_dynamics` driver handles this
automatically; manual use requires explicit calls.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..math import quaternions as _q
from ._common import (
    RowContribs,
    _add_block,
    assemble_rows,
    dot2_row,
    gamma_dot1,
)
from .base import Constraint, JacobianBlock
from .functions import Function

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["PrismaticDriver", "RevoluteDriver"]

_EX = np.array([1.0, 0.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])


class RevoluteDriver(Constraint):
    r"""Prescribes the rotation angle between two markers (1 equation).

    The angle :math:`\theta(t)` is measured between the local x-arms of the
    two markers about their common z-axis (the revolute axis).  The constraint
    equation is

    .. math::

        \mathbf u_{x,i} \cdot \mathbf u_{x,j} - \cos(\theta(t)) = 0,

    where :math:`\mathbf u_{x,i} = A_i\,\mathbf x'_{i}` and similarly for
    :math:`j`.

    .. warning::

        The cosine formulation is non-singular for
        :math:`\theta \in (0, \pi)` but degenerates near
        :math:`\theta \in \{0, \pi\}`.  For oscillatory motion confined
        to this range the constraint is well-conditioned.

    Parameters
    ----------
    marker_i, marker_j
        The two markers; their local z-axes define the revolute axis and
        their local x-axes are the angle-measurement arms.
    function
        Prescribed angle :math:`\theta(t)` [rad] as a
        :class:`~modyn.constraints.functions.Function`.
    name
        Optional human-readable label.
    """

    n_eq: int = 1

    def __init__(
        self,
        marker_i: Marker,
        marker_j: Marker,
        function: Function,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j
        self._fn = function
        self._x_i_body = marker_i.orientation_local @ _EX
        self._x_j_body = marker_j.orientation_local @ _EX
        self._t: float = 0.0

    def set_time(self, t: float) -> None:
        """Inject the current simulation time."""
        self._t = float(t)

    def _make_row(self) -> RowContribs:
        bi, bj = self._mi.parent, self._mj.parent
        u_xi = bi.rotation_matrix @ self._x_i_body
        u_xj = bj.rotation_matrix @ self._x_j_body

        f = self._fn.value(self._t)
        fd = self._fn.derivative(self._t)
        fdd = self._fn.second_derivative(self._t)
        cos_f = math.cos(f)
        sin_f = math.sin(f)

        phi = float(u_xi @ u_xj) - cos_f

        cross_ij = np.cross(u_xi, u_xj)
        Jp_i = cross_ij @ (2.0 * _q.E(bi.quaternion))
        Jp_j = -(cross_ij @ (2.0 * _q.E(bj.quaternion)))

        blocks: dict = {}
        _add_block(blocks, bi, np.zeros(3), Jp_i)
        _add_block(blocks, bj, np.zeros(3), Jp_j)

        # gamma = gamma_dot1 + d²(cos θ)/dt² = gamma_dot1 - cos(f)·ḟ² - sin(f)·f̈
        g_dot1 = gamma_dot1(bi.omega_global, u_xi, bj.omega_global, u_xj)
        gamma = g_dot1 - cos_f * fd**2 - sin_f * fdd

        return RowContribs(phi=phi, gamma=gamma, blocks=blocks)

    def phi(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[2]


class PrismaticDriver(Constraint):
    r"""Prescribes the displacement along the joint axis between two markers.

    The scalar displacement is

    .. math::

        d(t) = \mathbf u_{z,i} \cdot (\mathbf r_j^P - \mathbf r_i^P),

    where :math:`\mathbf u_{z,i} = A_i\,\mathbf z'_i` is the joint axis
    (z-axis of ``marker_i``) in the global frame and
    :math:`\mathbf r^P` are the global marker-origin positions.

    Use alongside :class:`~modyn.constraints.PrismaticJoint` (which
    removes the 4 orthogonal/rotational DOFs).

    Parameters
    ----------
    marker_i, marker_j
        Joint markers; the z-axis of ``marker_i`` defines the sliding axis.
    function
        Prescribed displacement :math:`d(t)` [m].
    name
        Optional human-readable label.
    """

    n_eq: int = 1

    def __init__(
        self,
        marker_i: Marker,
        marker_j: Marker,
        function: Function,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j
        self._fn = function
        self._z_i_body = marker_i.orientation_local @ _EZ
        self._t: float = 0.0

    def set_time(self, t: float) -> None:
        """Inject the current simulation time."""
        self._t = float(t)

    def _make_row(self) -> RowContribs:
        row_base = dot2_row(self._mi, self._z_i_body, self._mi, self._mj)
        f = self._fn.value(self._t)
        fdd = self._fn.second_derivative(self._t)
        phi_d = float(row_base.phi) - f
        gamma_d = float(row_base.gamma) + fdd
        return RowContribs(phi=phi_d, gamma=gamma_d, blocks=row_base.blocks)

    def phi(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[2]
