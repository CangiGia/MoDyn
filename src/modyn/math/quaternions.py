r"""Euler-parameter (unit-quaternion) algebra.

This module implements the **primary** orientation representation of
MoDyn. A body's orientation is stored as a 4-vector

.. math::

    \mathbf p = [e_0,\ e_1,\ e_2,\ e_3]^\mathsf{T}, \qquad
    \mathbf p^\mathsf{T} \mathbf p = 1.

The scalar component is :math:`e_0` (Shabana / Hamilton convention).

Conventions
-----------
- ``A(p)`` rotates **body-frame vectors into the global frame**:
  ``r_global = A(p) @ r_body``  (Shabana eq. 2.96).
- ``E(p)`` is the 3×4 matrix giving the **global** angular velocity:
  :math:`\\boldsymbol\\omega = 2\\,E(\\mathbf p)\\,\\dot{\\mathbf p}`
  (Shabana eq. 2.103).
- ``G(p)`` is the 3×4 matrix giving the **body** angular velocity:
  :math:`\\boldsymbol\\omega' = 2\\,G(\\mathbf p)\\,\\dot{\\mathbf p}`
  (Shabana eq. 2.107).
- Quaternion product follows the Hamilton convention; if
  ``p3 = qmul(p1, p2)`` then ``A(p3) == A(p1) @ A(p2)``.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., Wiley, 2010 — §2.6–2.8.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .vectors import skew

__all__ = [
    "A",
    "E",
    "G",
    "as_quat",
    "conjugate",
    "from_axis_angle",
    "identity",
    "norm",
    "normalize",
    "omega_body_to_pdot",
    "omega_to_pdot",
    "pdot_to_omega",
    "pdot_to_omega_body",
    "qmul",
    "rotate",
    "to_axis_angle",
]


# ─── Construction & basic algebra ─────────────────────────────────────


def as_quat(p: ArrayLike) -> NDArray[np.float64]:
    """Coerce ``p`` to a 1-D ``float64`` array of length 4.

    Does **not** normalize. Use :func:`normalize` explicitly when needed.
    """
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.size != 4:
        raise ValueError(f"expected a 4-vector quaternion, got shape {np.shape(p)}")
    return a.copy()


def identity() -> NDArray[np.float64]:
    """Identity quaternion ``[1, 0, 0, 0]`` (no rotation)."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def norm(p: ArrayLike) -> float:
    """Euclidean norm of a quaternion."""
    return float(np.linalg.norm(as_quat(p)))


def normalize(p: ArrayLike, *, tol: float = 1e-12) -> NDArray[np.float64]:
    """Return ``p / ||p||``.

    Raises
    ------
    ValueError
        If ``||p|| < tol`` (the input is not a valid rotation).
    """
    a = as_quat(p)
    n = float(np.linalg.norm(a))
    if n < tol:
        raise ValueError(f"cannot normalize a near-zero quaternion (||p|| = {n:.3e})")
    return a / n


def conjugate(p: ArrayLike) -> NDArray[np.float64]:
    """Quaternion conjugate ``[e0, -e1, -e2, -e3]``.

    For a unit quaternion this equals the inverse rotation.
    """
    a = as_quat(p)
    return np.array([a[0], -a[1], -a[2], -a[3]], dtype=np.float64)


def qmul(p: ArrayLike, q: ArrayLike) -> NDArray[np.float64]:
    """Hamilton quaternion product ``p ⊗ q``.

    Composition rule: if both ``p`` and ``q`` are unit, then
    ``A(qmul(p, q)) == A(p) @ A(q)``.
    """
    a, b = as_quat(p), as_quat(q)
    a0, av = a[0], a[1:]
    b0, bv = b[0], b[1:]
    out = np.empty(4, dtype=np.float64)
    out[0] = a0 * b0 - av @ bv
    out[1:] = a0 * bv + b0 * av + np.cross(av, bv)
    return out


def rotate(p: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """Rotate the 3-vector ``v`` (body frame) into the global frame.

    Equivalent to ``A(p) @ v`` but computed via the sandwich product.
    """
    a = as_quat(p)
    a0, av = a[0], a[1:]
    vv = np.asarray(v, dtype=np.float64).reshape(3)
    t = 2.0 * np.cross(av, vv)
    return vv + a0 * t + np.cross(av, t)


# ─── Axis-angle conversions ───────────────────────────────────────────


def from_axis_angle(axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Build a unit quaternion from an axis (need not be unit) and angle [rad]."""
    u = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(u))
    if n == 0.0:
        return identity()
    half = 0.5 * float(angle)
    s = np.sin(half) / n
    return np.array([np.cos(half), u[0] * s, u[1] * s, u[2] * s], dtype=np.float64)


def to_axis_angle(p: ArrayLike) -> tuple[NDArray[np.float64], float]:
    """Decompose a unit quaternion into ``(axis, angle)``.

    The returned axis is unit-norm; if the rotation is the identity the
    axis defaults to ``[1, 0, 0]`` and the angle is ``0``.
    """
    a = normalize(p)
    e0 = float(np.clip(a[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(e0)
    s = np.sqrt(max(1.0 - e0 * e0, 0.0))
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    return a[1:] / s, float(angle)


# ─── Rotation and transformation matrices ─────────────────────────────


def A(p: ArrayLike) -> NDArray[np.float64]:
    r"""Rotation matrix ``A(p)`` mapping body → global.

    Shabana eq. (2.96):

    .. math::

        A(\mathbf p) = (2 e_0^2 - 1)\,\mathbf I
        + 2\,(\mathbf e\,\mathbf e^\mathsf{T} + e_0\,\tilde{\mathbf e}),
        \quad \mathbf e = [e_1, e_2, e_3]^\mathsf{T}.
    """
    a = as_quat(p)
    e0 = a[0]
    e = a[1:]
    I3 = np.eye(3)
    return (2.0 * e0 * e0 - 1.0) * I3 + 2.0 * (np.outer(e, e) + e0 * skew(e))


def E(p: ArrayLike) -> NDArray[np.float64]:
    r"""3×4 matrix ``E(p)`` for the **global-frame** angular velocity.

    Shabana eq. (2.103):
    :math:`E = [\,-\mathbf e \;\; e_0 \mathbf I + \tilde{\mathbf e}\,]`,
    so :math:`\boldsymbol\omega = 2\,E\,\dot{\mathbf p}` and
    :math:`E\,E^\mathsf{T} = \mathbf I`, :math:`E\,\mathbf p = \mathbf 0`.
    """
    a = as_quat(p)
    e0 = a[0]
    e = a[1:]
    out = np.empty((3, 4), dtype=np.float64)
    out[:, 0] = -e
    out[:, 1:] = e0 * np.eye(3) + skew(e)
    return out


def G(p: ArrayLike) -> NDArray[np.float64]:
    r"""3×4 matrix ``G(p)`` for the **body-frame** angular velocity.

    Shabana eq. (2.107):
    :math:`G = [\,-\mathbf e \;\; e_0 \mathbf I - \tilde{\mathbf e}\,]`,
    so :math:`\boldsymbol\omega' = 2\,G\,\dot{\mathbf p}` and
    :math:`G\,G^\mathsf{T} = \mathbf I`, :math:`G\,\mathbf p = \mathbf 0`.
    """
    a = as_quat(p)
    e0 = a[0]
    e = a[1:]
    out = np.empty((3, 4), dtype=np.float64)
    out[:, 0] = -e
    out[:, 1:] = e0 * np.eye(3) - skew(e)
    return out


# ─── Angular-velocity ↔ ṗ conversions ─────────────────────────────────
#
# Inversion identities for unit p (Shabana §2.8):
#     E E^T = I,  E p = 0  ⇒  ṗ = (1/2) E^T ω + λ p
#     G G^T = I,  G p = 0  ⇒  ṗ = (1/2) G^T ω' + λ p
# The component along p is fixed by p^T ṗ = 0 (norm preservation): we
# project it out so the returned ṗ is consistent with a unit quaternion.


def _project_out_p(pdot: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
    return pdot - float(p @ pdot) * p


def pdot_to_omega(p: ArrayLike, pdot: ArrayLike) -> NDArray[np.float64]:
    """Global angular velocity from quaternion rate: ``ω = 2 E(p) ṗ``."""
    return 2.0 * E(p) @ as_quat(pdot)


def omega_to_pdot(p: ArrayLike, omega: ArrayLike) -> NDArray[np.float64]:
    """Quaternion rate from global angular velocity: ``ṗ = (1/2) E(p)^T ω``.

    The returned ``ṗ`` is projected onto the tangent space of the unit
    sphere (``p^T ṗ = 0``), so ``||p|| = 1`` is preserved to first order.
    """
    a = as_quat(p)
    pdot = 0.5 * E(a).T @ np.asarray(omega, dtype=np.float64).reshape(3)
    return _project_out_p(pdot, a)


def pdot_to_omega_body(p: ArrayLike, pdot: ArrayLike) -> NDArray[np.float64]:
    """Body angular velocity from quaternion rate: ``ω' = 2 G(p) ṗ``."""
    return 2.0 * G(p) @ as_quat(pdot)


def omega_body_to_pdot(p: ArrayLike, omega_body: ArrayLike) -> NDArray[np.float64]:
    """Quaternion rate from body angular velocity: ``ṗ = (1/2) G(p)^T ω'``.

    The result is projected onto the tangent space ``p^T ṗ = 0``.
    """
    a = as_quat(p)
    pdot = 0.5 * G(a).T @ np.asarray(omega_body, dtype=np.float64).reshape(3)
    return _project_out_p(pdot, a)
