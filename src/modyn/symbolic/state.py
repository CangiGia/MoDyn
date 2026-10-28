"""Symbolic view of the body state (positions and quasi-velocities)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca

from ..model.ground import Ground
from .quat import A_sx

if TYPE_CHECKING:
    from ..dynamics.coordinates import SystemCoordinates


class SymbolicState:
    r"""Symbolic accessors for body poses and velocities.

    Parameters
    ----------
    layout
        Coordinate layout of the system.
    q
        ``SX`` column of the position coordinates, shape ``(7n, 1)``.
    v
        ``SX`` column of the quasi-velocities :math:`[\dot{\mathbf R}, \boldsymbol\omega']`,
        shape ``(6n, 1)``.

    The ground is handled transparently: zero position, identity rotation,
    zero velocity.
    """

    def __init__(self, layout: SystemCoordinates, q, v) -> None:
        self.layout = layout
        self.q = q
        self.v = v
        self._A: dict[int, ca.SX] = {}

    @staticmethod
    def _is_ground(body) -> bool:
        return isinstance(body, Ground)

    def _index(self, body) -> int:
        return self.layout.offset(body) // 7

    def R(self, body):
        """Centroid position (3x1)."""
        if self._is_ground(body):
            return ca.SX.zeros(3, 1)
        s = self.layout.offset(body)
        return self.q[s : s + 3]

    def p(self, body):
        """Euler parameters (4x1); the identity quaternion for the ground."""
        if self._is_ground(body):
            return ca.SX([1, 0, 0, 0])
        s = self.layout.offset(body)
        return self.q[s + 3 : s + 7]

    def A(self, body):
        """Rotation matrix (3x3), cached per body."""
        if self._is_ground(body):
            return ca.SX.eye(3)
        key = id(body)
        if key not in self._A:
            self._A[key] = A_sx(self.p(body))
        return self._A[key]

    def Rdot(self, body):
        """Centroid velocity in the global frame (3x1)."""
        if self._is_ground(body):
            return ca.SX.zeros(3, 1)
        k = 6 * self._index(body)
        return self.v[k : k + 3]

    def omega_body(self, body):
        """Angular velocity in the body frame (3x1)."""
        if self._is_ground(body):
            return ca.SX.zeros(3, 1)
        k = 6 * self._index(body)
        return self.v[k + 3 : k + 6]

    def omega_global(self, body):
        """Angular velocity in the global frame (3x1)."""
        return ca.mtimes(self.A(body), self.omega_body(body))

    def point(self, body, s_local):
        """Global position of the body-fixed point ``s_local`` (numeric 3-vector)."""
        return self.R(body) + ca.mtimes(self.A(body), ca.DM(s_local))

    def point_velocity(self, body, s_local):
        """Global velocity of the body-fixed point ``s_local``."""
        r = ca.mtimes(self.A(body), ca.DM(s_local))
        return self.Rdot(body) + ca.cross(self.omega_global(body), r)
