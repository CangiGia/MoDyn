r"""Symbolic constraint kernels (coincidence, dot-1, dot-2).

Same equations as :func:`modyn.constraints._common.coincidence_rows`,
``dot1_row`` and ``dot2_row``, written on ``SX`` so that Jacobians come from
automatic differentiation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from ..model.marker import Marker
    from .state import SymbolicState

__all__ = ["coincidence", "dot1", "dot2"]


def coincidence(st: SymbolicState, mi: Marker, mj: Marker):
    r"""Three rows :math:`\mathbf r_i^P - \mathbf r_j^P`."""
    return st.point(mi.parent, mi.position_local) - st.point(mj.parent, mj.position_local)


def dot1(st: SymbolicState, ma: Marker, va_body, mb: Marker, vb_body):
    r"""One row :math:`(A_a\mathbf v'_a)^\mathsf{T}(A_b\mathbf v'_b)`."""
    ua = ca.mtimes(st.A(ma.parent), ca.DM(np.asarray(va_body, dtype=float)))
    ub = ca.mtimes(st.A(mb.parent), ca.DM(np.asarray(vb_body, dtype=float)))
    return ca.mtimes(ua.T, ub)


def dot2(st: SymbolicState, m_axis: Marker, v_axis_body, mi: Marker, mj: Marker):
    r"""One row :math:`(A_a\mathbf v'_a)^\mathsf{T}(\mathbf r_j^P - \mathbf r_i^P)`."""
    ua = ca.mtimes(st.A(m_axis.parent), ca.DM(np.asarray(v_axis_body, dtype=float)))
    d = st.point(mj.parent, mj.position_local) - st.point(mi.parent, mi.position_local)
    return ca.mtimes(ua.T, d)
