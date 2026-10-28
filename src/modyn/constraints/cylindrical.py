r"""Cylindrical joint.

The cylindrical joint allows two relative degrees of freedom along a
common axis: translation **and** rotation about the same axis. It is the
prismatic joint with the rotation-lock equation removed.

Joint axis convention
---------------------
The common axis is the **z-axis** (third column) of each marker's local
orientation.

Constraint equations (4 total)
------------------------------

1. :math:`\mathbf x_i \cdot \mathbf z_j = 0` — axes parallel.
2. :math:`\mathbf y_i \cdot \mathbf z_j = 0` — axes parallel.
3. :math:`\mathbf x_i \cdot \mathbf d   = 0` — no translation ⟂ to axis.
4. :math:`\mathbf y_i \cdot \mathbf d   = 0` — no translation ⟂ to axis.

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

__all__ = ["CylindricalJoint"]

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])


class CylindricalJoint(Constraint):
    """Translation + rotation about a common z-axis between two markers."""

    n_eq: int = 4

    def __init__(self, marker_i: Marker, marker_j: Marker, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j
        self._x_i_body = marker_i.orientation_local @ _EX
        self._y_i_body = marker_i.orientation_local @ _EY
        self._z_j_body = marker_j.orientation_local @ _EZ

    @property
    def marker_i(self) -> Marker:
        """Marker on body ``i``."""
        return self._mi

    @property
    def marker_j(self) -> Marker:
        """Marker on body ``j``."""
        return self._mj

    def _rows(self):
        return [
            dot1_row(self._mi, self._x_i_body, self._mj, self._z_j_body),
            dot1_row(self._mi, self._y_i_body, self._mj, self._z_j_body),
            dot2_row(self._mi, self._x_i_body, self._mi, self._mj),
            dot2_row(self._mi, self._y_i_body, self._mi, self._mj),
        ]

    def phi(self) -> NDArray[np.float64]:
        """Stack of the 4 scalar residuals."""
        return assemble_rows(self._rows(), self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:
        """Per-body Jacobian blocks (4 rows each)."""
        return assemble_rows(self._rows(), self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:
        """Acceleration-level RHS for the 4 equations."""
        return assemble_rows(self._rows(), self.n_eq)[2]
