"""Singleton ``Ground`` body — fixed inertial reference frame.

The ground exposes the same geometric API as :class:`RigidBody` (so that
markers and joints can treat it uniformly) but is permanently fixed at the
origin with identity orientation and zero velocity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.base import Base
from ..core.utils import ensure_shape
from ..math import quaternions as _q

__all__ = ["Ground"]


class Ground(Base):
    """Fixed inertial frame, exposed as a singleton.

    Use :meth:`instance` to retrieve the unique :class:`Ground` object.
    Direct instantiation is allowed but yields the same shared singleton.
    """

    _instance: Ground | None = None

    def __new__(cls) -> Ground:
        """Return the singleton ground instance, allocating it if needed."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise the singleton; idempotent across repeated calls."""
        if getattr(self, "_initialized", False):
            return
        super().__init__(name="ground")
        self._R: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._p: NDArray[np.float64] = _q.identity()
        self._zero3: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._initialized = True

    @classmethod
    def instance(cls) -> Ground:
        """Return the unique :class:`Ground` instance."""
        return cls()

    # ── Geometric API mirroring RigidBody (read-only) ─────────────────

    @property
    def mass(self) -> float:
        """Mass of the ground (``+∞`` by convention)."""
        return float("inf")

    @property
    def position(self) -> NDArray[np.float64]:
        """Origin of the inertial frame (always zero)."""
        return self._R

    @property
    def quaternion(self) -> NDArray[np.float64]:
        """Identity orientation as a unit quaternion."""
        return self._p

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """Identity rotation matrix."""
        return np.eye(3, dtype=np.float64)

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Linear velocity (always zero)."""
        return self._zero3

    @property
    def omega_body(self) -> NDArray[np.float64]:
        """Body-frame angular velocity (always zero)."""
        return self._zero3

    @property
    def omega_global(self) -> NDArray[np.float64]:
        """Global-frame angular velocity (always zero)."""
        return self._zero3

    def point_global(self, s_body: ArrayLike) -> NDArray[np.float64]:
        """Return ``s_body`` unchanged: ground points coincide with their local coordinates."""
        return ensure_shape(s_body, (3,), "s_body").copy()

    def velocity_of_point(self, s_body: ArrayLike) -> NDArray[np.float64]:
        """Return zero: every ground-fixed point has zero velocity."""
        ensure_shape(s_body, (3,), "s_body")
        return self._zero3.copy()
