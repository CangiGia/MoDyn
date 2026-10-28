r"""Detection and removal of redundant constraint rows.

A planar mechanism modelled in 3D, e.g. a four-bar linkage with four parallel
revolute joints, has more constraint equations than needed: the rows of the
Jacobian :math:`\mathbf D = \boldsymbol\Phi_q\mathbf T` are linearly dependent
and the saddle-point matrix of the equations of motion is singular.

:func:`select_independent_rows` picks a maximal set of linearly independent
rows with a rank-revealing QR factorization with column pivoting applied to
:math:`\mathbf D^\mathsf{T}`. The rows that are left out are, by the rank
condition, implied by the kept ones and can be removed from the DAE; their
multipliers are not unique (statically indeterminate reactions) and are
recovered afterwards in the minimum-norm sense.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import qr

__all__ = ["RowSelection", "select_independent_rows"]


class RowSelection(NamedTuple):
    """Outcome of the redundancy analysis.

    Attributes
    ----------
    rows
        Sorted indices of the kept (independent) rows.
    dropped
        Sorted indices of the removed (redundant) rows.
    rank
        Numerical rank of the Jacobian.
    """

    rows: NDArray[np.intp]
    dropped: NDArray[np.intp]
    rank: int

    @property
    def n_redundant(self) -> int:
        """Number of removed rows."""
        return int(self.dropped.size)


def select_independent_rows(D: ArrayLike, *, tol: float = 1e-9) -> RowSelection:
    r"""Select a maximal set of linearly independent rows of ``D``.

    Parameters
    ----------
    D
        Constraint Jacobian, shape ``(m, n)``, evaluated on an assembled
        configuration.
    tol
        Relative threshold on the diagonal of the triangular factor used to
        declare a row dependent. Rows are scaled to unit norm first, so the
        test is insensitive to the units of the equations.
    """
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    m = D.shape[0]
    if m == 0:
        empty = np.zeros(0, dtype=np.intp)
        return RowSelection(empty, empty, 0)

    norms = np.linalg.norm(D, axis=1)
    live = np.flatnonzero(norms > 1e-14)
    if live.size == 0:
        return RowSelection(
            np.zeros(0, dtype=np.intp), np.arange(m, dtype=np.intp), 0
        )

    Dn = D[live] / norms[live, None]
    _, R, piv = qr(Dn.T, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > tol * diag[0]))

    kept = np.sort(live[piv[:rank]]).astype(np.intp)
    dropped = np.setdiff1d(np.arange(m), kept).astype(np.intp)
    return RowSelection(kept, dropped, rank)
