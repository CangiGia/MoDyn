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

from ._common import (
    RowContribs,
    _add_block,
    assemble_rows,
    dot1_row,
    dot2_row,
)
from .base import Constraint, JacobianBlock
from .functions import Function

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["PrismaticDriver", "RevoluteDriver"]

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])


class RevoluteDriver(Constraint):
    r"""Prescribes the rotation angle between two markers (1 equation).

    The angle :math:`\theta(t)` is the rotation of marker ``j`` with respect
    to marker ``i`` about their common z-axis (the revolute axis). Writing
    :math:`\mathbf u_{x,j} = \cos\theta\,\mathbf u_{x,i} + \sin\theta\,\mathbf u_{y,i}`,
    the constraint equation is

    .. math::

        \sin\theta(t)\,(\mathbf u_{x,i}\cdot\mathbf u_{x,j})
        - \cos\theta(t)\,(\mathbf u_{y,i}\cdot\mathbf u_{x,j}) = 0,

    where :math:`\mathbf u_{\bullet,i} = A_i\,\mathbf a'_{\bullet,i}`. The left
    side equals :math:`\sin(\theta(t) - \theta_{rel})`, so the equation is
    smooth for any angle (including full revolutions) and its Jacobian never
    vanishes near the solution.

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
        self._y_i_body = marker_i.orientation_local @ _EY
        self._x_j_body = marker_j.orientation_local @ _EX
        self._t: float = 0.0

    def set_time(self, t: float) -> None:
        """Inject the current simulation time."""
        self._t = float(t)

    def _make_row(self) -> RowContribs:
        bi, bj = self._mi.parent, self._mj.parent
        r1 = dot1_row(self._mi, self._x_i_body, self._mj, self._x_j_body)
        r2 = dot1_row(self._mi, self._y_i_body, self._mj, self._x_j_body)

        f = self._fn.value(self._t)
        fd = self._fn.derivative(self._t)
        fdd = self._fn.second_derivative(self._t)
        s, c = math.sin(f), math.cos(f)
        sd, cd = c * fd, -s * fd
        sdd, cdd = c * fdd - s * fd**2, -s * fdd - c * fd**2

        # Time derivatives of the two dot-1 kernels from the angular velocities.
        u_xi = bi.rotation_matrix @ self._x_i_body
        u_yi = bi.rotation_matrix @ self._y_i_body
        u_xj = bj.rotation_matrix @ self._x_j_body
        wi, wj = bi.omega_global, bj.omega_global
        d1 = np.cross(wi, u_xi) @ u_xj + u_xi @ np.cross(wj, u_xj)
        d2 = np.cross(wi, u_yi) @ u_xj + u_yi @ np.cross(wj, u_xj)

        phi = s * float(r1.phi) - c * float(r2.phi)
        gamma = (
            s * float(r1.gamma)
            - c * float(r2.gamma)
            - (2.0 * (sd * d1 - cd * d2) + sdd * float(r1.phi) - cdd * float(r2.phi))
        )

        blocks: dict = {}
        for body, (J_R, J_p) in r1.blocks.items():
            _add_block(blocks, body, s * J_R, s * J_p)
        for body, (J_R, J_p) in r2.blocks.items():
            _add_block(blocks, body, -c * J_R, -c * J_p)
        return RowContribs(phi=phi, gamma=gamma, blocks=blocks)

    def phi(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[2]

    def phi_sym(self, state, t):  # noqa: D102
        import casadi as ca

        from ..symbolic import dot1

        f = self._fn.value_sym(t)
        d1 = dot1(state, self._mi, self._x_i_body, self._mj, self._x_j_body)
        d2 = dot1(state, self._mi, self._y_i_body, self._mj, self._x_j_body)
        return ca.sin(f) * d1 - ca.cos(f) * d2


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

    def phi_sym(self, state, t):  # noqa: D102
        from ..symbolic import dot2

        return dot2(state, self._mi, self._z_i_body, self._mi, self._mj) - self._fn.value_sym(t)

    def phi(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([self._make_row()], self.n_eq)[2]
