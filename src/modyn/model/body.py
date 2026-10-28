"""Spatial rigid body.

State (per body, 7 generalized coordinates):

- centroid position ``R ∈ ℝ³`` (global frame),
- orientation ``p ∈ ℝ⁴`` (unit Euler-parameter quaternion).

Velocity is stored as 6 components: linear velocity ``Ṙ`` and angular
velocity — by default in the **body frame** (``ω'``), which makes the
inertia tensor constant. The user can also drive/read global-frame angular
velocity ``ω`` via :attr:`omega_global`.

Reference: Shabana, *Computational Dynamics* 3rd ed., chapter 6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.base import Base
from ..core.utils import (
    ensure_finite,
    ensure_positive,
    ensure_shape,
    ensure_symmetric_positive_definite,
)
from ..math import quaternions as _q

__all__ = ["RigidBody"]


class RigidBody(Base):
    """A spatial rigid body in absolute coordinates.

    Parameters
    ----------
    mass
        Total mass, must be strictly positive.
    inertia
        3×3 inertia tensor expressed in the body frame, must be SPD.
    position
        Initial centroid position in the global frame, default origin.
    quaternion
        Initial unit quaternion (orientation, body → global), default identity.
    velocity
        Initial linear velocity (global frame), default zero.
    omega_body
        Initial angular velocity expressed in the body frame, default zero.
    name
        Optional human-readable name.

    Notes
    -----
    The body never owns its joints/forces; those live on the topology side
    of the model. A body only owns its inertial properties and current state.
    """

    def __init__(
        self,
        mass: float,
        inertia: ArrayLike,
        *,
        position: ArrayLike = (0.0, 0.0, 0.0),
        quaternion: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        velocity: ArrayLike = (0.0, 0.0, 0.0),
        omega_body: ArrayLike = (0.0, 0.0, 0.0),
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self._mass: float = ensure_positive(mass, "mass")
        self._inertia: NDArray[np.float64] = ensure_symmetric_positive_definite(
            ensure_shape(inertia, (3, 3), "inertia"), "inertia"
        )
        self._R: NDArray[np.float64] = ensure_finite(
            ensure_shape(position, (3,), "position"), "position"
        ).copy()
        self._p: NDArray[np.float64] = _q.normalize(quaternion)
        self._Rdot: NDArray[np.float64] = ensure_finite(
            ensure_shape(velocity, (3,), "velocity"), "velocity"
        ).copy()
        self._omega_body: NDArray[np.float64] = ensure_finite(
            ensure_shape(omega_body, (3,), "omega_body"), "omega_body"
        ).copy()

    # ── Inertial properties (read-only) ───────────────────────────────

    @property
    def mass(self) -> float:
        """Total mass of the body."""
        return self._mass

    @property
    def inertia(self) -> NDArray[np.float64]:
        """Inertia tensor in the body frame (constant)."""
        return self._inertia

    @property
    def inertia_global(self) -> NDArray[np.float64]:
        """Inertia tensor expressed in the global frame: ``A J A^T``."""
        A = _q.A(self._p)
        return A @ self._inertia @ A.T

    # ── Generalized coordinates ───────────────────────────────────────

    @property
    def position(self) -> NDArray[np.float64]:
        """Centroid position in the global frame."""
        return self._R

    @position.setter
    def position(self, value: ArrayLike) -> None:
        self._R = ensure_finite(ensure_shape(value, (3,), "position"), "position").copy()

    @property
    def quaternion(self) -> NDArray[np.float64]:
        """Orientation as a unit Euler-parameter quaternion."""
        return self._p

    @quaternion.setter
    def quaternion(self, value: ArrayLike) -> None:
        self._p = _q.normalize(value)

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """3×3 rotation matrix corresponding to ``self.quaternion``."""
        return _q.A(self._p)

    @property
    def q(self) -> NDArray[np.float64]:
        """7-vector ``[R, p]`` of generalized coordinates."""
        return np.concatenate((self._R, self._p))

    @q.setter
    def q(self, value: ArrayLike) -> None:
        v = ensure_shape(value, (7,), "q")
        self.position = v[:3]
        self.quaternion = v[3:]

    def set_q(self, value: ArrayLike, *, normalize: bool = True) -> None:
        r"""Set the 7-vector ``[R, p]`` with optional quaternion normalization.

        Parameters
        ----------
        value
            Generalized coordinates ``[R, p]``, shape ``(7,)``.
        normalize
            When ``True`` (default) the quaternion part is renormalized to unit
            norm. Pass ``False`` to store the raw Euler parameters verbatim, as
            required by an index-3 integrator that treats
            :math:`\mathbf p^\mathsf{T}\mathbf p - 1 = 0` as an explicit
            constraint with its own Lagrange multiplier.
        """
        v = ensure_shape(value, (7,), "q")
        self.position = v[:3]
        if normalize:
            self.quaternion = v[3:]
        else:
            self._p = ensure_finite(ensure_shape(v[3:], (4,), "quaternion"), "quaternion").copy()

    # ── Velocities ────────────────────────────────────────────────────

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Linear velocity of the centroid in the global frame."""
        return self._Rdot

    @velocity.setter
    def velocity(self, value: ArrayLike) -> None:
        self._Rdot = ensure_finite(ensure_shape(value, (3,), "velocity"), "velocity").copy()

    @property
    def omega_body(self) -> NDArray[np.float64]:
        """Angular velocity expressed in the body frame."""
        return self._omega_body

    @omega_body.setter
    def omega_body(self, value: ArrayLike) -> None:
        self._omega_body = ensure_finite(
            ensure_shape(value, (3,), "omega_body"), "omega_body"
        ).copy()

    @property
    def omega_global(self) -> NDArray[np.float64]:
        """Angular velocity expressed in the global frame: ``A ω'``."""
        return _q.A(self._p) @ self._omega_body

    @omega_global.setter
    def omega_global(self, value: ArrayLike) -> None:
        w = ensure_finite(ensure_shape(value, (3,), "omega_global"), "omega_global")
        self._omega_body = _q.A(self._p).T @ w

    # ── Geometric helpers ─────────────────────────────────────────────

    def point_global(self, s_body: ArrayLike) -> NDArray[np.float64]:
        """Map a body-frame point ``s'`` to the global frame: ``R + A s'``."""
        s = ensure_shape(s_body, (3,), "s_body")
        return self._R + _q.rotate(self._p, s)

    def velocity_of_point(self, s_body: ArrayLike) -> NDArray[np.float64]:
        """Global-frame velocity of a body-fixed point ``s'``.

        ``v_P = Ṙ + ω × (A s')``.
        """
        s = ensure_shape(s_body, (3,), "s_body")
        r_global = _q.rotate(self._p, s)
        return self._Rdot + np.cross(self.omega_global, r_global)
