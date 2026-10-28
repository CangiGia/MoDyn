r"""Primitive constraints: standalone building blocks for custom joints.

Each class wraps one of the low-level kernel rows from
:mod:`modyn.constraints._common` as an independent
:class:`~modyn.constraints.base.Constraint` subclass.  They are intended
for users who want to assemble custom joints or add individual constraint
rows without using the pre-built joint types.

Available primitives
--------------------
:class:`CoincidenceConstraint`
    3 equations — forces two marker origins to coincide.
    Equivalent to :class:`~modyn.constraints.SphericalJoint`.
:class:`Dot1Constraint`
    1 equation — enforces orthogonality of two body-fixed vectors
    :math:`\mathbf u_a^\mathsf{T}\mathbf u_b = 0`.
:class:`Dot2Constraint`
    1 equation — enforces orthogonality of a body-fixed vector to the
    relative position vector between two marker origins
    :math:`\mathbf u_a^\mathsf{T}(\mathbf r_j^P - \mathbf r_i^P) = 0`.

Reference: Haug, *Computer Aided Kinematics and Dynamics*, ch. 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._common import assemble_rows, coincidence_rows, dot1_row, dot2_row
from .base import Constraint, JacobianBlock

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["CoincidenceConstraint", "Dot1Constraint", "Dot2Constraint"]


class CoincidenceConstraint(Constraint):
    r"""Three-equation constraint enforcing :math:`\mathbf r_i^P = \mathbf r_j^P`.

    Equivalent to :class:`~modyn.constraints.SphericalJoint`.

    Parameters
    ----------
    marker_i, marker_j
        The two markers whose global origins must coincide.
    name
        Optional label.
    """

    n_eq: int = 3

    def __init__(self, marker_i: Marker, marker_j: Marker, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j

    def phi(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([coincidence_rows(self._mi, self._mj)], self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:  # noqa: D102
        return assemble_rows([coincidence_rows(self._mi, self._mj)], self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([coincidence_rows(self._mi, self._mj)], self.n_eq)[2]

    def phi_sym(self, state, t):  # noqa: D102
        from ..symbolic import coincidence

        return coincidence(state, self._mi, self._mj)


class Dot1Constraint(Constraint):
    r"""One-equation orthogonality constraint :math:`\mathbf u_a \cdot \mathbf u_b = 0`.

    Parameters
    ----------
    marker_a
        Marker on body :math:`a`; its parent owns the first vector.
    v_a_body
        Body-frame vector :math:`\mathbf v'_a` on ``marker_a.parent``,
        shape ``(3,)``.
    marker_b
        Marker on body :math:`b`; its parent owns the second vector.
    v_b_body
        Body-frame vector :math:`\mathbf v'_b` on ``marker_b.parent``,
        shape ``(3,)``.
    name
        Optional label.
    """

    n_eq: int = 1

    def __init__(
        self,
        marker_a: Marker,
        v_a_body: ArrayLike,
        marker_b: Marker,
        v_b_body: ArrayLike,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._ma = marker_a
        self._mb = marker_b
        self._va = np.asarray(v_a_body, dtype=np.float64)
        self._vb = np.asarray(v_b_body, dtype=np.float64)

    def phi(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([dot1_row(self._ma, self._va, self._mb, self._vb)], self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:  # noqa: D102
        return assemble_rows([dot1_row(self._ma, self._va, self._mb, self._vb)], self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([dot1_row(self._ma, self._va, self._mb, self._vb)], self.n_eq)[2]

    def phi_sym(self, state, t):  # noqa: D102
        from ..symbolic import dot1

        return dot1(state, self._ma, self._va, self._mb, self._vb)


class Dot2Constraint(Constraint):
    r"""One-equation constraint :math:`\mathbf u_a \cdot (\mathbf r_j^P - \mathbf r_i^P) = 0`.

    The axis vector :math:`\mathbf u_a = A_a\,\mathbf v'_a` is body-fixed
    on ``marker_axis.parent``.

    Parameters
    ----------
    marker_axis
        Marker whose parent body carries the axis vector.
    v_axis_body
        Axis vector in the body frame of ``marker_axis.parent``, shape ``(3,)``.
    marker_pt_i, marker_pt_j
        Markers defining the relative-position vector
        :math:`\mathbf d = \mathbf r_j^P - \mathbf r_i^P`.
    name
        Optional label.
    """

    n_eq: int = 1

    def __init__(
        self,
        marker_axis: Marker,
        v_axis_body: ArrayLike,
        marker_pt_i: Marker,
        marker_pt_j: Marker,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._m_ax = marker_axis
        self._m_i = marker_pt_i
        self._m_j = marker_pt_j
        self._v = np.asarray(v_axis_body, dtype=np.float64)

    def phi(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([dot2_row(self._m_ax, self._v, self._m_i, self._m_j)], self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:  # noqa: D102
        return assemble_rows([dot2_row(self._m_ax, self._v, self._m_i, self._m_j)], self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:  # noqa: D102
        return assemble_rows([dot2_row(self._m_ax, self._v, self._m_i, self._m_j)], self.n_eq)[2]

    def phi_sym(self, state, t):  # noqa: D102
        from ..symbolic import dot2

        return dot2(state, self._m_ax, self._v, self._m_i, self._m_j)
