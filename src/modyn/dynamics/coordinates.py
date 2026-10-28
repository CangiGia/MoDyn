r"""System-level generalized-coordinate bookkeeping.

A multibody system in MoDyn uses **absolute coordinates**: every moving
body owns the 7-vector :math:`\mathbf q_B = [\mathbf R_B^\mathsf{T},\,
\mathbf p_B^\mathsf{T}]^\mathsf{T}` (centroid position + unit quaternion).
Stacking the bodies in a fixed order gives the global coordinate vector

.. math::

    \mathbf q = [\mathbf q_1^\mathsf{T}, \dots, \mathbf q_{n}^\mathsf{T}]^\mathsf{T}
    \in \mathbb R^{7n},

and likewise the global velocity vector
:math:`\dot{\mathbf q} = [\dot{\mathbf R}_B^\mathsf{T},\, \dot{\mathbf p}_B^\mathsf{T}]`.

:class:`SystemCoordinates` is a thin, stateless-by-design helper that maps
an ordered set of :class:`~modyn.model.body.RigidBody` objects to their
slices in the global vectors and gathers / scatters those vectors to and
from the bodies. The :class:`~modyn.model.ground.Ground` carries no
coordinates and is never part of the layout.

Reference: Shabana, *Computational Dynamics*, 3rd ed., §6.1–6.2.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.utils import ensure_shape
from ..math import quaternions as _q
from ..model.body import RigidBody
from ..model.ground import Ground

__all__ = ["SystemCoordinates"]

#: Generalized coordinates contributed by one rigid body: ``[R(3), p(4)]``.
N_PER_BODY: int = 7


class SystemCoordinates:
    r"""Index map between rigid bodies and the global ``q`` / ``q̇`` vectors.

    Parameters
    ----------
    bodies
        Ordered iterable of moving bodies. The order fixes the block layout
        of the global vectors and is preserved. Each body must be a
        :class:`~modyn.model.body.RigidBody`; passing a
        :class:`~modyn.model.ground.Ground` raises :class:`ValueError`
        (the ground has no generalized coordinates).

    Raises
    ------
    ValueError
        If ``bodies`` is empty, contains a :class:`Ground`, or contains the
        same body twice.
    TypeError
        If ``bodies`` contains a non-:class:`RigidBody`.

    Notes
    -----
    Each body occupies a contiguous block of :data:`N_PER_BODY` ``= 7``
    entries, ordered ``[R_x, R_y, R_z, e_0, e_1, e_2, e_3]`` in ``q`` and
    ``[Ṙ_x, Ṙ_y, Ṙ_z, ė_0, ė_1, ė_2, ė_3]`` in ``q̇``.
    """

    def __init__(self, bodies: Iterable[RigidBody]) -> None:
        ordered = tuple(bodies)
        if not ordered:
            raise ValueError("SystemCoordinates requires at least one body")

        seen: set[int] = set()
        for k, body in enumerate(ordered):
            if isinstance(body, Ground):
                raise ValueError("Ground has no generalized coordinates and cannot be laid out")
            if not isinstance(body, RigidBody):
                raise TypeError(f"bodies[{k}] is {type(body).__name__}, expected RigidBody")
            if id(body) in seen:
                raise ValueError(f"body {body.name!r} appears more than once in the layout")
            seen.add(id(body))

        self._bodies: tuple[RigidBody, ...] = ordered
        self._offset_of: dict[int, int] = {id(b): k * N_PER_BODY for k, b in enumerate(ordered)}

    # ── Layout ─────────────────────────────────────────────────────────

    @property
    def bodies(self) -> tuple[RigidBody, ...]:
        """The ordered bodies, in global-vector block order."""
        return self._bodies

    @property
    def n_bodies(self) -> int:
        """Number of moving bodies in the layout."""
        return len(self._bodies)

    @property
    def n_coords(self) -> int:
        """Dimension of the global ``q`` / ``q̇`` vectors, ``7 · n_bodies``."""
        return N_PER_BODY * len(self._bodies)

    def offset(self, body: RigidBody) -> int:
        """Return the start index of ``body``'s block in the global vector."""
        try:
            return self._offset_of[id(body)]
        except KeyError:
            raise KeyError(f"body {body.name!r} is not part of this layout") from None

    def slice(self, body: RigidBody) -> slice:
        """Return the ``slice`` selecting ``body``'s block in the global vector."""
        start = self.offset(body)
        return slice(start, start + N_PER_BODY)

    # ── Gather (bodies → global vector) ───────────────────────────────

    def assemble_q(self) -> NDArray[np.float64]:
        r"""Gather the global position vector :math:`\mathbf q` from the bodies."""
        q = np.empty(self.n_coords, dtype=np.float64)
        for body in self._bodies:
            q[self.slice(body)] = body.q
        return q

    def assemble_qdot(self) -> NDArray[np.float64]:
        r"""Gather the global velocity vector :math:`\dot{\mathbf q}`.

        The block layout is :math:`[\dot{\mathbf R},\, \dot{\mathbf p}]` per
        body. The quaternion rate is reconstructed from each body's stored
        body-frame angular velocity via
        :func:`~modyn.math.quaternions.omega_body_to_pdot`.
        """
        qd = np.empty(self.n_coords, dtype=np.float64)
        for body in self._bodies:
            start = self.offset(body)
            qd[start : start + 3] = body.velocity
            qd[start + 3 : start + N_PER_BODY] = _q.omega_body_to_pdot(
                body.quaternion, body.omega_body
            )
        return qd

    # ── Scatter (global vector → bodies) ──────────────────────────────

    def scatter_q(self, q: ArrayLike, *, normalize: bool = True) -> None:
        r"""Distribute a global position vector :math:`\mathbf q` onto the bodies.

        Parameters
        ----------
        q
            Global position vector, shape ``(7n,)``.
        normalize
            When ``True`` (default) each body's quaternion is renormalized to
            unit norm on assignment. Pass ``False`` to store the raw Euler
            parameters verbatim, as required by an index-3 integrator that
            enforces :math:`\mathbf p^\mathsf{T}\mathbf p - 1 = 0` explicitly
            through its own Lagrange multiplier.
        """
        q_arr = ensure_shape(q, (self.n_coords,), "q")
        for body in self._bodies:
            body.set_q(q_arr[self.slice(body)], normalize=normalize)

    def scatter_qdot(self, qdot: ArrayLike) -> None:
        r"""Distribute a global velocity vector :math:`\dot{\mathbf q}` onto the bodies.

        The linear part sets :attr:`RigidBody.velocity`; the quaternion-rate
        part is converted to a body-frame angular velocity via
        :func:`~modyn.math.quaternions.pdot_to_omega_body` and stored on
        :attr:`RigidBody.omega_body`. The conversion uses each body's
        **current** quaternion, so call :meth:`scatter_q` first when setting a
        full state.
        """
        qd_arr = ensure_shape(qdot, (self.n_coords,), "qdot")
        for body in self._bodies:
            start = self.offset(body)
            body.velocity = qd_arr[start : start + 3]
            body.omega_body = _q.pdot_to_omega_body(
                body.quaternion, qd_arr[start + 3 : start + N_PER_BODY]
            )

    def set_state(self, q: ArrayLike, qdot: ArrayLike) -> None:
        """Scatter positions then velocities (in that order) onto the bodies."""
        self.scatter_q(q)
        self.scatter_qdot(qdot)

    # ── Dunder ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of bodies in the layout."""
        return len(self._bodies)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Return ``<SystemCoordinates n_bodies=… n_coords=…>``."""
        return f"<SystemCoordinates n_bodies={self.n_bodies} n_coords={self.n_coords}>"
