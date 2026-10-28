"""User-defined generalized-force callback."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Force

if TYPE_CHECKING:
    from ..dynamics.coordinates import SystemCoordinates

__all__ = ["UserForce"]


class UserForce(Force):
    """Force defined by an arbitrary user callback.

    Parameters
    ----------
    fn
        ``fn(layout, t) -> ndarray(7n,)`` — returns the full generalized-force
        contribution for all bodies given the current layout and time.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def generalized_force(
        self,
        layout: SystemCoordinates,
        t: float,
    ) -> NDArray[np.float64]:
        """Evaluate the user callback and return the result, shape ``(7n,)``."""
        return np.asarray(self._fn(layout, t), dtype=np.float64).reshape(layout.n_coords)
