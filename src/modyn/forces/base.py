"""Abstract base class for all generalized-force objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..dynamics.coordinates import SystemCoordinates

__all__ = ["Force"]


class Force(ABC):
    r"""Abstract base for generalized forces acting on system bodies.

    Subclasses compute their contribution to the global generalized-force
    vector :math:`\mathbf Q_e` given the current state stored on the bodies
    in ``layout``. The caller (e.g. :func:`~modyn.solver.dynamics.integrate_dynamics`)
    scatters the trial state before invoking this method, so body positions
    and velocities are already up to date.
    """

    @abstractmethod
    def generalized_force(
        self,
        layout: SystemCoordinates,
        t: float,
    ) -> NDArray[np.float64]:
        r"""Return the generalized-force contribution, shape ``(7n,)``.

        Parameters
        ----------
        layout
            Coordinate layout holding the current body state.
        t
            Current simulation time.
        """

    def wrench_sym(self, state, t):
        r"""Symbolic contribution as a list of ``(body, F, tau_body)``.

        ``F`` is the force on the centroid in the global frame and
        ``tau_body`` the torque about the centroid in the body frame, both
        CasADi ``SX`` columns. These are exactly the components of the force
        on the quasi-velocities :math:`[\dot{\mathbf R}, \boldsymbol\omega']`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no symbolic formulation (wrench_sym)"
        )
