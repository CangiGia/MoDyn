"""External / internal forces and torques.

All force objects subclass :class:`Force` and expose
``generalized_force(layout, t) -> ndarray(7n,)``.
:func:`to_force_fn` converts a list of :class:`Force` objects into the
callable expected by :func:`~modyn.solver.dynamics.integrate_dynamics`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .applied import PointForce
from .base import Force
from .gravity import Gravity
from .springs import TSDA
from .torque import Torque
from .user import UserForce

if TYPE_CHECKING:
    from ..dynamics.coordinates import SystemCoordinates

__all__ = ["TSDA", "Force", "Gravity", "PointForce", "Torque", "UserForce", "to_force_fn"]


def to_force_fn(
    forces: list[Force],
    layout: SystemCoordinates,
):
    """Convert a list of :class:`Force` objects into a dynamics-compatible callable.

    Parameters
    ----------
    forces
        List of :class:`Force` instances to sum.
    layout
        Coordinate layout; captured by closure.

    Returns
    -------
    callable
        ``fn(t, q, qdot) -> ndarray(7n,)`` suitable as the ``applied_forces``
        argument of :func:`~modyn.solver.dynamics.integrate_dynamics`.  The
        callable reads body state from ``layout`` — the bodies must already
        have been scattered, which the integrator does automatically.
    """

    def fn(
        t: float,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        total = np.zeros(layout.n_coords, dtype=np.float64)
        for f in forces:
            total += f.generalized_force(layout, t)
        return total

    return fn
