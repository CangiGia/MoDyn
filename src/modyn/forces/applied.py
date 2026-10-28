r"""Point force applied at a marker."""

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
    from ..model.marker import Marker

__all__ = ["PointForce"]


class PointForce(Force):
    r"""Force applied at a marker, expressed in the **global** frame.

    The generalized-force contribution follows from virtual work
    :math:`\delta W = \mathbf F^\mathsf{T}\,\delta\mathbf r_P` where
    :math:`\mathbf r_P = \mathbf R + \mathbf A \bar{\mathbf s}`:

    .. math::

        (\mathbf Q_e)_R = \mathbf F, \qquad
        (\mathbf Q_e)_p = 2\,G^\mathsf{T}\,
            (\bar{\mathbf s} \times \mathbf A^\mathsf{T}\mathbf F)

    where :math:`\bar{\mathbf s}` is the body-frame marker position and
    :math:`G` is the body-frame angular-velocity matrix.

    Parameters
    ----------
    marker
        Attachment point. Its parent body must appear in the layout.
    force
        Force in the global frame [N]. Either a constant array ``(3,)`` or
        a callable ``f(t) -> ndarray(3,)``.
    """

    def __init__(
        self,
        marker: Marker,
        force: ArrayLike | Callable[[float], NDArray[np.float64]],
    ) -> None:
        self._marker = marker
        if callable(force):
            self._force_fn: Callable[[float], NDArray[np.float64]] = force
        else:
            _f: NDArray[np.float64] = ensure_finite(
                ensure_shape(force, (3,), "force"), "force"
            ).copy()
            self._force_fn = lambda t: _f

    @property
    def marker(self) -> Marker:
        """Attachment marker."""
        return self._marker

    def generalized_force(
        self,
        layout: SystemCoordinates,
        t: float,
    ) -> NDArray[np.float64]:
        r"""Compute ``Q_R = F`` and ``Q_p = 2 G^T (s' × A^T F)``."""
        Qe = np.zeros(layout.n_coords, dtype=np.float64)
        F = np.asarray(self._force_fn(t), dtype=np.float64)
        body = self._marker.parent
        try:
            s = layout.offset(body)
        except KeyError:
            return Qe  # force on Ground — no DOFs
        p = body.quaternion
        G = _q.G(p)  # (3, 4)
        A = _q.A(p)  # (3, 3)
        s_body = self._marker.position_local  # (3,)
        tau_body = np.cross(s_body, A.T @ F)  # torque in body frame
        Qe[s : s + 3] = F
        Qe[s + 3 : s + 7] = 2.0 * G.T @ tau_body
        return Qe
