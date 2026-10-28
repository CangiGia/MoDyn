r"""Universal (Hooke / Cardan) joint.

The universal joint links two bodies through a cross with two
perpendicular pins, allowing two rotational degrees of freedom while
constraining all translations and the relative rotation about the
"missing" axis.

Joint axis convention
---------------------
The **z-axis** (third column) of each marker is the pin axis on its body.
For a physically meaningful Hooke joint the two pin axes must be
perpendicular when the joint is assembled — i.e. the user must orient the
two markers so that :math:`\mathbf z_i\cdot\mathbf z_j = 0` in the initial
configuration. The constraint then keeps them perpendicular as the joint
articulates.

Constraint equations (4 total)
------------------------------

1. **Coincidence (3)** — :math:`\mathbf r_i^P - \mathbf r_j^P = \mathbf 0`.
2. **Pin perpendicularity (1)** — :math:`\mathbf z_i \cdot \mathbf z_j = 0`.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., §6.6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._common import assemble_rows, coincidence_rows, dot1_row
from .base import Constraint, JacobianBlock

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["UniversalJoint"]

_EZ = np.array([0.0, 0.0, 1.0])


class UniversalJoint(Constraint):
    """Hooke joint: coincidence + perpendicular pin axes."""

    n_eq: int = 4

    def __init__(self, marker_i: Marker, marker_j: Marker, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j
        self._z_i_body = marker_i.orientation_local @ _EZ
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
            coincidence_rows(self._mi, self._mj),
            dot1_row(self._mi, self._z_i_body, self._mj, self._z_j_body),
        ]

    def phi(self) -> NDArray[np.float64]:
        """Stack of coincidence (3) and pin perpendicularity (1)."""
        return assemble_rows(self._rows(), self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:
        """Per-body Jacobian blocks (4 rows each)."""
        return assemble_rows(self._rows(), self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:
        """Acceleration-level RHS for the 4 equations."""
        return assemble_rows(self._rows(), self.n_eq)[2]
