r"""Revolute joint (hinge / pin).

The revolute joint allows a single rotational degree of freedom about a
common axis. Built as a spherical-joint kernel (coincidence of the two
marker origins, 3 equations) plus two ``dot-1`` constraints enforcing
that the joint axis on body :math:`i` (the z-axis of ``marker_i``) stays
parallel to the joint axis on body :math:`j` (the z-axis of ``marker_j``).
Total: **5 equations**, **1 DoF**.

Joint axis convention
---------------------
The revolute axis is the **z-axis** (third column) of each marker's local
orientation. The two body-fixed vectors orthogonal to the joint axis on
body ``i`` — namely the x- and y- axes of ``marker_i`` — are dotted with
the j-side axis. Aligning both dot products to zero forces the two
z-axes to be parallel.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., §6.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._common import assemble_rows, coincidence_rows, dot1_row
from .base import Constraint, JacobianBlock

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["RevoluteJoint"]

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])


class RevoluteJoint(Constraint):
    r"""Hinge joint enforcing coaxial rotation between two markers.

    Constraint equations (5 total):

    1. **Coincidence (3)** — :math:`\mathbf r_i^P - \mathbf r_j^P = \mathbf 0`.
    2. **Axis alignment (2)** — :math:`x_i\cdot z_j = 0` and
       :math:`y_i\cdot z_j = 0`, with ``x_i, y_i`` the global axes of
       ``marker_i`` and ``z_j`` the global z-axis of ``marker_j``.

    The joint axis (z-axis of either marker, identical when satisfied) is
    the single remaining rotational DoF.

    Parameters
    ----------
    marker_i, marker_j
        The two markers defining the hinge; their z-axes are the hinge axis.
    name
        Optional human-readable name.
    """

    n_eq: int = 5

    def __init__(self, marker_i: Marker, marker_j: Marker, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j
        # Body-frame x/y axes on marker_i and z axis on marker_j.
        self._x_i_body = marker_i.orientation_local @ _EX
        self._y_i_body = marker_i.orientation_local @ _EY
        self._z_j_body = marker_j.orientation_local @ _EZ

    @property
    def marker_i(self) -> Marker:
        """Marker on body ``i`` (defines the ``+`` side of the residual)."""
        return self._mi

    @property
    def marker_j(self) -> Marker:
        """Marker on body ``j`` (defines the ``-`` side of the residual)."""
        return self._mj

    def _rows(self):
        return [
            coincidence_rows(self._mi, self._mj),
            dot1_row(self._mi, self._x_i_body, self._mj, self._z_j_body),
            dot1_row(self._mi, self._y_i_body, self._mj, self._z_j_body),
        ]

    def phi(self) -> NDArray[np.float64]:
        """Stack of coincidence (3) and two axis-alignment dot products (2)."""
        return assemble_rows(self._rows(), self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:
        """Per-body Jacobian blocks (5 rows each)."""
        return assemble_rows(self._rows(), self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:
        """Acceleration-level RHS for the 5 equations."""
        return assemble_rows(self._rows(), self.n_eq)[2]
