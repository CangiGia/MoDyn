r"""Spherical joint (ball-and-socket).

The spherical joint coincides the origins of two markers, eliminating
their relative translation while leaving all three relative rotations
free. It contributes three scalar equations.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., §6.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._common import assemble_rows, coincidence_rows
from .base import Constraint, JacobianBlock

if TYPE_CHECKING:
    from ..model.marker import Marker

__all__ = ["SphericalJoint"]


class SphericalJoint(Constraint):
    r"""Ball-and-socket joint coinciding two marker origins.

    Enforces

    .. math::

        \boldsymbol\Phi(q) = \mathbf r_i^P - \mathbf r_j^P = \mathbf 0,

    i.e. global-frame coincidence of the two marker origins. Three scalar
    equations; three translational DoFs removed, all relative rotations free.

    Parameters
    ----------
    marker_i, marker_j
        The two markers whose origins must coincide.
    name
        Optional human-readable name.
    """

    n_eq: int = 3

    def __init__(self, marker_i: Marker, marker_j: Marker, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._mi = marker_i
        self._mj = marker_j

    @property
    def marker_i(self) -> Marker:
        """First marker (defining the ``+`` side of the residual)."""
        return self._mi

    @property
    def marker_j(self) -> Marker:
        """Second marker (defining the ``-`` side of the residual)."""
        return self._mj

    # ── Constraint contract ───────────────────────────────────────────

    def _rows(self):
        return [coincidence_rows(self._mi, self._mj)]

    def phi(self) -> NDArray[np.float64]:
        """Residual ``r_i^P - r_j^P``."""
        return assemble_rows(self._rows(), self.n_eq)[0]

    def phi_q(self) -> list[JacobianBlock]:
        """Per-body Jacobian blocks (3 rows each)."""
        return assemble_rows(self._rows(), self.n_eq)[1]

    def gamma(self) -> NDArray[np.float64]:
        r"""Acceleration-level RHS for the 3 equations.

        .. math::

            \gamma = -\bigl[\boldsymbol\omega_i\!\times\!(\boldsymbol\omega_i\!\times\!
            \mathbf u_i) - \boldsymbol\omega_j\!\times\!(\boldsymbol\omega_j\!\times\!
            \mathbf u_j)\bigr].
        """
        return assemble_rows(self._rows(), self.n_eq)[2]
