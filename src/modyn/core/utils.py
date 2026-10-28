"""Generic, dependency-free validators and small helpers.

This module is imported by virtually every other MoDyn subpackage; it must
stay free of any internal dependency and have no side effects on import.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "as_float_array",
    "ensure_finite",
    "ensure_non_negative",
    "ensure_positive",
    "ensure_shape",
    "ensure_symmetric_positive_definite",
]


def as_float_array(x: ArrayLike, *, copy: bool = True) -> NDArray[np.float64]:
    """Return ``x`` as a contiguous ``float64`` ndarray."""
    a = np.asarray(x, dtype=np.float64)
    return a.copy() if copy else a


def ensure_shape(x: ArrayLike, shape: tuple[int, ...], name: str = "value") -> NDArray[np.float64]:
    """Validate that ``x`` has the given shape; return it as ``float64``."""
    a = as_float_array(x, copy=False)
    if a.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    return a


def ensure_finite(x: ArrayLike, name: str = "value") -> NDArray[np.float64]:
    """Validate that all entries of ``x`` are finite."""
    a = as_float_array(x, copy=False)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: contains non-finite entries")
    return a


def ensure_positive(value: float, name: str = "value") -> float:
    """Validate that ``value`` is a strictly positive finite scalar."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name}: expected a strictly positive scalar, got {value!r}")
    return v


def ensure_non_negative(value: float, name: str = "value") -> float:
    """Validate that ``value`` is a non-negative finite scalar."""
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name}: expected a non-negative scalar, got {value!r}")
    return v


def ensure_symmetric_positive_definite(
    M: ArrayLike, name: str = "matrix", *, atol: float = 1e-12
) -> NDArray[np.float64]:
    """Validate that ``M`` is a symmetric positive-definite square matrix.

    Returns the symmetrized matrix ``(M + M.T) / 2`` to absorb tiny round-off.
    """
    A = as_float_array(M, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: expected a square matrix, got shape {A.shape}")
    if not np.allclose(A, A.T, atol=atol):
        raise ValueError(f"{name}: not symmetric within atol={atol}")
    A = 0.5 * (A + A.T)
    eigs = np.linalg.eigvalsh(A)
    if eigs.min() <= 0.0:
        raise ValueError(f"{name}: not positive-definite (min eig = {eigs.min():.3e})")
    return A
