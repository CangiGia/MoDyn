r"""SE(3) homogeneous transforms.

A *transform* is a rigid motion ``T = (R, t)`` with ``R ∈ SO(3)`` and
``t ∈ ℝ^3``, mapping a body-frame point ``p_body`` to a global-frame
point as

.. math::

    \mathbf p_\text{global} = R\,\mathbf p_\text{body} + \mathbf t.

This convenience layer is meant for I/O, visualization and assembly
helpers. The dynamic solver works directly on ``(position, quaternion)``
pairs from :mod:`modyn.math.quaternions` to avoid building 4×4 matrices
in the inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import quaternions as _q
from . import rotations as _rot

__all__ = ["Transform", "from_homogeneous", "to_homogeneous"]


@dataclass(frozen=True, slots=True)
class Transform:
    """Immutable SE(3) transform stored as ``(quaternion, translation)``.

    Attributes
    ----------
    p : ndarray of shape (4,)
        Unit Euler-parameter quaternion (orientation, body → global).
    t : ndarray of shape (3,)
        Translation in the global frame.

    Notes
    -----
    Storing orientation as a quaternion (rather than a 3×3 matrix) keeps
    the representation singularity-free and consistent with body state.
    The 4×4 homogeneous matrix is built on demand via :func:`to_homogeneous`.
    """

    p: NDArray[np.float64]
    t: NDArray[np.float64]

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def identity(cls) -> Transform:
        """Identity transform (no rotation, zero translation)."""
        return cls(_q.identity(), np.zeros(3, dtype=np.float64))

    @classmethod
    def from_quat_translation(cls, p: ArrayLike, t: ArrayLike) -> Transform:
        """Build from a quaternion and a translation."""
        return cls(_q.normalize(p), np.asarray(t, dtype=np.float64).reshape(3).copy())

    @classmethod
    def from_matrix_translation(cls, R: ArrayLike, t: ArrayLike) -> Transform:
        """Build from a rotation matrix and a translation."""
        return cls(_rot.matrix_to_quat(R), np.asarray(t, dtype=np.float64).reshape(3).copy())

    @classmethod
    def from_homogeneous(cls, H: ArrayLike) -> Transform:
        """Build from a 4×4 homogeneous matrix."""
        return from_homogeneous(H)

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def R(self) -> NDArray[np.float64]:
        """3×3 rotation matrix corresponding to ``self.p``."""
        return _q.A(self.p)

    def as_matrix(self) -> NDArray[np.float64]:
        """4×4 homogeneous representation of this transform."""
        return to_homogeneous(self.p, self.t)

    # ── Algebra ───────────────────────────────────────────────────────

    def __matmul__(self, other: Transform) -> Transform:
        """Compose transforms: ``T_self @ T_other`` applies ``other`` first."""
        if not isinstance(other, Transform):
            return NotImplemented
        return Transform(
            _q.qmul(self.p, other.p),
            _q.rotate(self.p, other.t) + self.t,
        )

    def inverse(self) -> Transform:
        """Inverse rigid motion."""
        p_inv = _q.conjugate(self.p)
        return Transform(p_inv, -_q.rotate(p_inv, self.t))

    def apply(self, v: ArrayLike) -> NDArray[np.float64]:
        """Apply this transform to a 3-vector point."""
        return _q.rotate(self.p, np.asarray(v, dtype=np.float64).reshape(3)) + self.t


def to_homogeneous(p: ArrayLike, t: ArrayLike) -> NDArray[np.float64]:
    """Build the 4×4 homogeneous matrix from a quaternion and a translation."""
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = _q.A(p)
    H[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return H


def from_homogeneous(H: ArrayLike) -> Transform:
    """Inverse of :func:`to_homogeneous`."""
    M = np.asarray(H, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"expected a 4×4 matrix, got shape {M.shape}")
    return Transform(_rot.matrix_to_quat(M[:3, :3]), M[:3, 3].copy())
