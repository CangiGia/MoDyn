r"""Pure torque applied to a body."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.utils import ensure_finite, ensure_shape
from ..math import quaternions as _q
from .base import Force

if TYPE_CHECKING:
    from ..dynamics.coordinates import SystemCoordinates
    from ..model.body import RigidBody

__all__ = ["Torque"]


class Torque(Force):
    r"""Torque applied to a body, expressed in the **global** frame.

    The torque in the body frame is :math:`\boldsymbol\tau' = A^\mathsf{T}\boldsymbol\tau`
    and the generalized force on the Euler parameters is
    :math:`(\mathbf Q_e)_p = 2\,G^\mathsf{T}\boldsymbol\tau'`.

    Parameters
    ----------
    body
        Body receiving the torque.
    torque
        Torque in the global frame [N m]: a constant ``(3,)`` array or a
        callable ``f(t) -> (3,)``.
    """

    def __init__(
        self,
        body: RigidBody,
        torque: ArrayLike | Callable[[float], NDArray[np.float64]],
    ) -> None:
        self._body = body
        if callable(torque):
            self._torque_fn: Callable[[float], NDArray[np.float64]] = torque
        else:
            _t = ensure_finite(ensure_shape(torque, (3,), "torque"), "torque").copy()
            self._torque_fn = lambda t: _t

    @property
    def body(self) -> RigidBody:
        """Body receiving the torque."""
        return self._body

    def generalized_force(self, layout: SystemCoordinates, t: float) -> NDArray[np.float64]:
        """Return ``Q_p = 2 G^T A^T tau`` on the body's Euler parameters."""
        Qe = np.zeros(layout.n_coords, dtype=np.float64)
        s = layout.offset(self._body)
        p = self._body.quaternion
        tau_body = _q.A(p).T @ np.asarray(self._torque_fn(t), dtype=np.float64)
        Qe[s + 3 : s + 7] = 2.0 * _q.G(p).T @ tau_body
        return Qe

    def wrench_sym(self, state, t):
        """Symbolic wrench: zero force and body-frame torque ``A^T tau``."""
        import casadi as ca

        tau = self._torque_fn(t)
        if not isinstance(tau, ca.SX | ca.DM):
            tau = ca.vertcat(*np.asarray(tau, dtype=object).ravel().tolist())
        return [(self._body, ca.SX.zeros(3, 1), ca.mtimes(state.A(self._body).T, tau))]
