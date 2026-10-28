"""Body-fixed reference frame (point + orientation).

A :class:`Marker` is the canonical attachment point for joints and forces.
It stores its position ``s'`` and orientation ``A_m`` in the **body frame**
of its owner. Global-frame quantities are derived on demand from the body
state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.base import Base
from ..core.utils import ensure_finite, ensure_shape
from ..math.rotations import is_rotation_matrix

if TYPE_CHECKING:
    from .body import RigidBody
    from .ground import Ground

__all__ = ["Marker"]


class Marker(Base):
    """Body-fixed frame attached to a parent body (or to ground).

    Parameters
    ----------
    parent
        The body owning this marker; may be a :class:`Ground` instance.
    position
        Marker origin expressed in the **parent body** frame, default origin.
    orientation
        3×3 rotation matrix mapping marker-frame vectors to body-frame
        vectors. Default is the identity.
    name
        Optional human-readable name.
    """

    def __init__(
        self,
        parent: RigidBody | Ground,
        *,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._parent = parent
        self._s_body: NDArray[np.float64] = ensure_finite(
            ensure_shape(position, (3,), "position"), "position"
        ).copy()
        if orientation is None:
            orientation = np.eye(3)
        A_m = ensure_shape(orientation, (3, 3), "orientation")
        if not is_rotation_matrix(A_m):
            raise ValueError("orientation: not a valid rotation matrix")
        self._A_m: NDArray[np.float64] = A_m.copy()

    # ── Parent / local properties ─────────────────────────────────────

    @property
    def parent(self) -> RigidBody | Ground:
        """Parent body (or :class:`Ground`) owning this marker."""
        return self._parent

    @property
    def position_local(self) -> NDArray[np.float64]:
        """Marker origin expressed in the parent body frame (constant)."""
        return self._s_body

    @property
    def orientation_local(self) -> NDArray[np.float64]:
        """Marker orientation in the parent body frame (constant 3×3)."""
        return self._A_m

    # ── Global properties (derived) ───────────────────────────────────

    @property
    def position_global(self) -> NDArray[np.float64]:
        """Marker origin in the global frame: ``R + A s'``."""
        return self._parent.point_global(self._s_body)

    @property
    def orientation_global(self) -> NDArray[np.float64]:
        """Marker orientation in the global frame: ``A_parent A_m``."""
        return self._parent.rotation_matrix @ self._A_m
