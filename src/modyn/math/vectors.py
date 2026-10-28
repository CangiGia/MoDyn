"""Vector helpers for 3D rigid-body kinematics.

All functions here are pure, allocation-light and operate on plain NumPy
arrays. They are the lowest layer of MoDyn: this module imports nothing
from the rest of the package.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., Wiley, 2010 — ch. 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_vec3",
    "cross",
    "skew",
    "unskew",
]


def as_vec3(v: ArrayLike) -> NDArray[np.float64]:
    """Coerce ``v`` to a 1-D ``float64`` array of length 3.

    Parameters
    ----------
    v : array_like
        Anything broadcastable to shape ``(3,)`` (list, tuple, ndarray of
        shape ``(3,)``, ``(3, 1)`` or ``(1, 3)``).

    Returns
    -------
    ndarray of shape (3,), dtype float64
        Independent copy — safe to mutate by callers.

    Raises
    ------
    ValueError
        If ``v`` does not contain exactly 3 elements.
    """
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"expected a 3-vector, got shape {np.shape(v)}")
    return a.copy()


def skew(v: ArrayLike) -> NDArray[np.float64]:
    r"""Skew-symmetric (tilde) matrix ``ṽ`` of a 3-vector.

    Defined so that ``ṽ @ w == cross(v, w)`` for every ``w ∈ ℝ^3``.

    .. math::

        \\tilde{\\mathbf v} =
        \\begin{bmatrix} 0 & -v_z & v_y \\\\ v_z & 0 & -v_x \\\\ -v_y & v_x & 0 \\end{bmatrix}

    Parameters
    ----------
    v : array_like, shape (3,)
        Input vector.

    Returns
    -------
    ndarray of shape (3, 3), dtype float64
        Skew-symmetric matrix.

    See Also
    --------
    unskew : inverse operation.
    """
    x, y, z = as_vec3(v)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def unskew(M: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`skew`: extract the 3-vector from a skew matrix.

    Parameters
    ----------
    M : array_like, shape (3, 3)
        Skew-symmetric matrix.

    Returns
    -------
    ndarray of shape (3,), dtype float64
        Vector ``v`` such that ``skew(v) == M``.

    Raises
    ------
    ValueError
        If ``M`` is not 3×3.

    Notes
    -----
    No symmetry check is performed: callers receive the average of the
    off-diagonal antisymmetric parts, which is the most numerically robust
    choice when ``M`` is only approximately skew (e.g. coming from
    finite-difference computations).
    """
    A = np.asarray(M, dtype=np.float64)
    if A.shape != (3, 3):
        raise ValueError(f"expected a 3×3 matrix, got shape {A.shape}")
    return 0.5 * np.array(
        [A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]],
        dtype=np.float64,
    )


def cross(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Right-handed cross product of two 3-vectors.

    Thin wrapper around :func:`numpy.cross` that enforces shape ``(3,)`` and
    ``float64`` dtype, matching the rest of this module's contract.
    """
    return np.cross(as_vec3(a), as_vec3(b))
