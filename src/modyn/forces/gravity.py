"""Uniform gravity field applied to every body in the layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.utils import ensure_finite, ensure_shape
from .base import Force

if TYPE_CHECKING:
    from ..dynamics.coordinates import SystemCoordinates

__all__ = ["Gravity"]


class Gravity(Force):
    r"""Uniform gravitational acceleration applied to all bodies.

    The force on each body is :math:`m \mathbf g` and acts at the centre of
    mass, so the generalized-force contribution is

    .. math::

        (\mathbf Q_e)_R = m\,\mathbf g, \qquad
        (\mathbf Q_e)_p = \mathbf 0.

    The zero :math:`p`-part follows because the torque about the centre of
    mass from a uniform field is zero.

    Parameters
    ----------
    g_vec
        Gravitational acceleration in the global frame, shape ``(3,)``,
        in m s⁻². Default ``(0, 0, -9.81)``.
    """

    def __init__(self, g_vec: ArrayLike = (0.0, 0.0, -9.81)) -> None:
        self._g: NDArray[np.float64] = ensure_finite(
            ensure_shape(g_vec, (3,), "g_vec"), "g_vec"
        ).copy()

    @property
    def g_vec(self) -> NDArray[np.float64]:
        """Gravity acceleration vector in the global frame (m s⁻²)."""
        return self._g

    def generalized_force(
        self,
        layout: SystemCoordinates,
        t: float,
    ) -> NDArray[np.float64]:
        """Return ``m*g`` for each body's translational DOFs; zero for orientation."""
        Qe = np.zeros(layout.n_coords, dtype=np.float64)
        for body in layout.bodies:
            s = layout.offset(body)
            Qe[s : s + 3] = body.mass * self._g
        return Qe
