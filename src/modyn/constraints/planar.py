r"""Planar joint (point-in-plane with normal lock).

The planar joint constrains body :math:`j` to slide on a plane fixed to
body :math:`i` (the plane through ``marker_i``'s origin, with normal
:math:`\mathbf z_i`). Three relative DoFs survive: two in-plane
translations and a rotation about the plane normal.

Joint axis convention
---------------------
The plane normal is the **z-axis** (third column) of each marker's local
orientation. The two markers must initially share the same plane normal
direction.

Constraint equations (3 total)
------------------------------

1. :math:`\mathbf x_i \cdot \mathbf z_j = 0` — plane normals parallel.
2. :math:`\mathbf y_i \cdot \mathbf z_j = 0` — plane normals parallel.
3. :math:`\mathbf z_i \cdot \mathbf d   = 0` — :math:`\mathbf r_j^P` lies in the plane.

with :math:`\mathbf d = \mathbf r_j^P - \mathbf r_i^P`.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., §6.6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._common import assemble_rows, dot1_row, dot2_row
from .base import Constraint, JacobianBlock

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["PlanarJoint"]

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])


class PlanarJoint(Constraint):
    """Point-in-plane with parallel-normal lock; 3 DoFs survive."""

    n_eq: int = 3

    def __init__(self, marker_i: Marker, marker_j: Marker, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j
        self._x_i_body = marker_i.orientation_local @ _EX
        self._y_i_body = marker_i.orientation_local @ _EY
        self._z_i_body = marker_i.orientation_local @ _EZ
        self._z_j_body = marker_j.orientation_local @ _EZ

    @property
    def marker_i(self) -> Marker:
        """Marker on body ``i`` (defines the plane)."""
        return self._mi

    @property
    def marker_j(self) -> Marker:
        """Marker on body ``j`` (the point constrained to lie on the plane)."""
        return self._mj

    def _rows(self):
        return [
            dot1_row(self._mi, self._x_i_body, self._mj, self._z_j_body),
            dot1_row(self._mi, self._y_i_body, self._mj, self._z_j_body),
            dot2_row(self._mi, self._z_i_body, self._mi, self._mj),
        ]

    def phi(self) -> NDArray[np.float64]:
        """Stack of the 3 scalar residuals."""
        return assemble_rows(self._rows(), self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:
        """Per-body Jacobian blocks (3 rows each)."""
        return assemble_rows(self._rows(), self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:
        """Acceleration-level RHS for the 3 equations."""
        return assemble_rows(self._rows(), self.n_eq)[2]
