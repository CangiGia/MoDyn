"""Rotation matrices and Euler/Cardan-angle conversions.

Euler/Cardan angles are supported only as **input/output utilities**: the
state representation of orientation is the unit quaternion (see
:mod:`modyn.math.quaternions`). Converters here exist for user
convenience (parsing input data, plotting outputs, debugging) and must
never be used inside the integration loop.

Convention
----------
- Angles are in radians.
- Sequences are *intrinsic* (rotations around the moving frame), expressed
  as 3-character strings of axis labels, e.g. ``"xyz"`` or ``"zxz"``.
- The resulting matrix maps **body → global** (consistent with
  :func:`modyn.math.quaternions.A`).

The supported sequences cover the 12 classical Euler/Cardan conventions:

- Cardan / Tait-Bryan : ``xyz, xzy, yxz, yzx, zxy, zyx``
- Proper Euler        : ``xyx, xzx, yxy, yzy, zxz, zyz``

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., Wiley, 2010 — §2.5–2.6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import quaternions as _q

__all__ = [
    "from_euler",
    "is_rotation_matrix",
    "matrix_to_quat",
    "quat_to_matrix",
    "rotx",
    "roty",
    "rotz",
    "to_euler",
]


# ─── Elementary rotations ─────────────────────────────────────────────


def rotx(theta: float) -> NDArray[np.float64]:
    """Rotation matrix about the x-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def roty(theta: float) -> NDArray[np.float64]:
    """Rotation matrix about the y-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotz(theta: float) -> NDArray[np.float64]:
    """Rotation matrix about the z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


_ELEM = {"x": rotx, "y": roty, "z": rotz}


def _validate_sequence(seq: str) -> str:
    s = seq.lower()
    if len(s) != 3 or any(c not in "xyz" for c in s):
        raise ValueError(f"invalid axis sequence {seq!r}; expected 3 chars from 'xyz'")
    if s[0] == s[1] or s[1] == s[2]:
        raise ValueError(f"invalid axis sequence {seq!r}; consecutive axes must differ")
    return s


# ─── Euler angles → matrix / quaternion ───────────────────────────────


def from_euler(seq: str, angles: ArrayLike) -> NDArray[np.float64]:
    """Build a rotation matrix from a 3-axis intrinsic angle sequence.

    Parameters
    ----------
    seq : str
        3-character axis sequence (see module docstring).
    angles : array_like, shape (3,)
        Angles in radians, in the order of ``seq``.

    Returns
    -------
    ndarray of shape (3, 3)
        Rotation matrix ``A = R(seq[0], a0) @ R(seq[1], a1) @ R(seq[2], a2)``.
    """
    s = _validate_sequence(seq)
    a = np.asarray(angles, dtype=np.float64).reshape(3)
    return _ELEM[s[0]](a[0]) @ _ELEM[s[1]](a[1]) @ _ELEM[s[2]](a[2])


def to_euler(seq: str, R: ArrayLike) -> NDArray[np.float64]:
    """Extract the angles of an intrinsic Euler/Cardan sequence from a matrix.

    Notes
    -----
    Each sequence has known singular configurations (gimbal lock):
    Cardan (e.g. ``xyz``) is singular at the middle angle ``±π/2``;
    proper Euler (e.g. ``zxz``) is singular at the middle angle ``0`` or
    ``±π``. At a singularity the split between the first and third angle
    is conventionally absorbed into the first.
    """
    s = _validate_sequence(seq)
    M = np.asarray(R, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"expected a 3×3 matrix, got shape {M.shape}")

    # General-purpose extractor based on Shoemake (1985), generalized to
    # arbitrary 3-axis sequences. Indices below follow the axis labels.
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    i, j, k = (axis_to_idx[c] for c in s)
    proper = s[0] == s[2]  # proper Euler vs Cardan / Tait-Bryan
    # Sign of the off-diagonal terms (handedness of the (i,j,k) triplet).
    sign = 1.0 if (j - i) % 3 == 1 else -1.0

    if proper:
        # Third repeated axis = i; the missing axis is the one not in (i, j).
        m = ({0, 1, 2} - {i, j}).pop()
        sy = np.hypot(M[i, j], M[i, m])
        if sy > 1e-12:
            a0 = np.arctan2(M[j, i], -sign * M[m, i])
            a1 = np.arctan2(sy, M[i, i])
            a2 = np.arctan2(M[i, j], sign * M[i, m])
        else:  # singular: a1 = 0 or π
            a0 = np.arctan2(-sign * M[j, m], M[j, j])
            a1 = np.arctan2(sy, M[i, i])
            a2 = 0.0
    else:  # Cardan / Tait-Bryan
        sy = np.hypot(M[i, i], M[i, j])
        if sy > 1e-12:
            a0 = np.arctan2(-sign * M[j, k], M[k, k])
            a1 = np.arctan2(sign * M[i, k], sy)
            a2 = np.arctan2(-sign * M[i, j], M[i, i])
        else:  # singular: a1 = ±π/2
            a0 = np.arctan2(-sign * M[j, i], M[j, j])
            a1 = np.arctan2(sign * M[i, k], sy)
            a2 = 0.0

    return np.array([a0, a1, a2], dtype=np.float64)


# ─── Matrix ↔ quaternion ──────────────────────────────────────────────


def quat_to_matrix(p: ArrayLike) -> NDArray[np.float64]:
    """Alias of :func:`modyn.math.quaternions.A` for symmetry of API."""
    return _q.A(p)


def matrix_to_quat(R: ArrayLike) -> NDArray[np.float64]:
    """Convert a rotation matrix to a unit quaternion.

    Uses Shepperd's method (numerically stable for any input rotation).
    The returned quaternion has non-negative scalar component to remove
    the ``±p`` sign ambiguity.
    """
    M = np.asarray(R, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"expected a 3×3 matrix, got shape {M.shape}")

    tr = M[0, 0] + M[1, 1] + M[2, 2]
    candidates = np.array(
        [
            1.0 + tr,
            1.0 + 2.0 * M[0, 0] - tr,
            1.0 + 2.0 * M[1, 1] - tr,
            1.0 + 2.0 * M[2, 2] - tr,
        ]
    )
    k = int(np.argmax(candidates))
    s = np.sqrt(max(candidates[k], 0.0))  # clamp tiny negatives from FP
    if s < 1e-12:
        return _q.identity()

    if k == 0:
        e0 = 0.5 * s
        inv = 0.5 / s
        e1 = (M[2, 1] - M[1, 2]) * inv
        e2 = (M[0, 2] - M[2, 0]) * inv
        e3 = (M[1, 0] - M[0, 1]) * inv
    elif k == 1:
        e1 = 0.5 * s
        inv = 0.5 / s
        e0 = (M[2, 1] - M[1, 2]) * inv
        e2 = (M[1, 0] + M[0, 1]) * inv
        e3 = (M[0, 2] + M[2, 0]) * inv
    elif k == 2:
        e2 = 0.5 * s
        inv = 0.5 / s
        e0 = (M[0, 2] - M[2, 0]) * inv
        e1 = (M[1, 0] + M[0, 1]) * inv
        e3 = (M[2, 1] + M[1, 2]) * inv
    else:
        e3 = 0.5 * s
        inv = 0.5 / s
        e0 = (M[1, 0] - M[0, 1]) * inv
        e1 = (M[0, 2] + M[2, 0]) * inv
        e2 = (M[2, 1] + M[1, 2]) * inv

    p = np.array([e0, e1, e2, e3], dtype=np.float64)
    if p[0] < 0:
        p = -p
    return p / np.linalg.norm(p)


# ─── Validation helpers ───────────────────────────────────────────────


def is_rotation_matrix(R: ArrayLike, *, tol: float = 1e-9) -> bool:
    """Check that ``R`` is orthogonal with determinant ``+1``."""
    M = np.asarray(R, dtype=np.float64)
    if M.shape != (3, 3):
        return False
    if not np.allclose(M.T @ M, np.eye(3), atol=tol):
        return False
    return abs(np.linalg.det(M) - 1.0) < tol
