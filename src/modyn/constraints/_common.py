r"""Shared low-level helpers used by all joint implementations.

This module exposes three building blocks reused by every lower-pair
joint:

- :func:`coincidence_rows` — 3 rows enforcing two marker origins to
  coincide; the spherical-joint kernel.
- :func:`dot1_row` — 1 row enforcing the orthogonality of two body-fixed
  vectors :math:`\mathbf u_a^\mathsf{T}\mathbf u_b = 0`.
- :func:`dot2_row` — 1 row enforcing the orthogonality of a body-fixed
  vector to the position vector between two marker origins,
  :math:`\mathbf u_a^\mathsf{T}(\mathbf r_j^P - \mathbf r_i^P) = 0`.

Each primitive returns a :class:`RowContribs` carrying the residual, the
per-body Jacobian rows and the γ contribution. The function
:func:`assemble_rows` stacks a list of :class:`RowContribs` into the final
``(phi, blocks, gamma)`` tuple expected by
:class:`~modyn.constraints.base.Constraint`.

Not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..math import quaternions as _q
from ..math.vectors import skew
from .base import JacobianBlock

if TYPE_CHECKING:
    from ..model.body import RigidBody
    from ..model.ground import Ground
    from ..model.marker import Marker

    BodyLike = RigidBody | Ground

__all__ = [
    "RowContribs",
    "assemble_rows",
    "centripetal",
    "coincidence_rows",
    "dA_dp",
    "dot1_row",
    "dot2_row",
    "gamma_dot1",
    "has_dofs",
]


# ─── Low-level math helpers ───────────────────────────────────────────


def has_dofs(body: BodyLike) -> bool:
    """Return ``True`` iff ``body`` exposes free DoFs (i.e. is not a ground).

    Ground sets ``mass = +inf``; rigid bodies have finite mass.
    """
    return bool(np.isfinite(getattr(body, "mass", np.inf)))


def dA_dp(u: ArrayLike, p: ArrayLike) -> NDArray[np.float64]:
    r"""Return :math:`\partial(A\,\mathbf s')/\partial \mathbf p`, shape ``(3, 4)``.

    Given a body-frame vector :math:`\mathbf s'` rotated into the global
    frame as :math:`\mathbf u = A(\mathbf p)\,\mathbf s'`, the derivative
    with respect to the body's Euler parameters is

    .. math::

        \frac{\partial \mathbf u}{\partial \mathbf p}
        = -\,\widetilde{\mathbf u}\;\bigl(2\,E(\mathbf p)\bigr).
    """
    return -skew(u) @ (2.0 * _q.E(p))


def centripetal(omega: ArrayLike, u: ArrayLike) -> NDArray[np.float64]:
    r"""Return :math:`\boldsymbol\omega \times (\boldsymbol\omega \times \mathbf u)`."""
    w = np.asarray(omega, dtype=np.float64)
    v = np.asarray(u, dtype=np.float64)
    return np.cross(w, np.cross(w, v))


def gamma_dot1(
    w_a: NDArray[np.float64],
    u_a: NDArray[np.float64],
    w_b: NDArray[np.float64],
    u_b: NDArray[np.float64],
) -> float:
    r"""γ for a dot-1 row :math:`\Phi = \mathbf u_a \cdot \mathbf u_b`.

    .. math::

        \gamma = -\Bigl[\boldsymbol\omega_a\!\times\!(\boldsymbol\omega_a\!\times\!
        \mathbf u_a)\cdot \mathbf u_b
        + 2\,(\boldsymbol\omega_a\!\times\!\mathbf u_a)\cdot
        (\boldsymbol\omega_b\!\times\!\mathbf u_b)
        + \mathbf u_a\cdot \boldsymbol\omega_b\!\times\!(\boldsymbol\omega_b\!\times\!
        \mathbf u_b)\Bigr].
    """
    cwa_ua = np.cross(w_a, u_a)
    cwb_ub = np.cross(w_b, u_b)
    return -float(
        centripetal(w_a, u_a) @ u_b + 2.0 * (cwa_ua @ cwb_ub) + u_a @ centripetal(w_b, u_b)
    )


# ─── Row contributions and assembler ─────────────────────────────────


class RowContribs(NamedTuple):
    """All contributions to one constraint row (or stack of consecutive rows).

    Attributes
    ----------
    phi
        Residual value(s): ``float`` for a single scalar row, or
        ``ndarray(k,)`` for a block of ``k`` rows.
    gamma
        Acceleration-level RHS, with the same shape convention as ``phi``.
    blocks
        Per-body Jacobian contribution. Keys are bodies; values are
        ``(J_R, J_p)`` with shapes ``(3,)/(4,)`` for a single row or
        ``(k, 3)/(k, 4)`` for a ``k``-row stack. Ground bodies may be
        present and will be filtered out by :func:`assemble_rows`.
    """

    phi: float | NDArray[np.float64]
    gamma: float | NDArray[np.float64]
    blocks: dict[BodyLike, tuple[NDArray[np.float64], NDArray[np.float64]]]


def assemble_rows(
    rows: list[RowContribs], n_eq: int
) -> tuple[NDArray[np.float64], list[JacobianBlock], NDArray[np.float64]]:
    """Stack :class:`RowContribs` into ``(phi, blocks, gamma)``.

    Each entry in ``rows`` contributes either 1 row or ``k`` consecutive
    rows (depending on the shape of its ``phi``). Per-body Jacobians are
    accumulated and zero-padded so each affected body's block has shape
    ``(n_eq, 3)`` / ``(n_eq, 4)``. Ground bodies are skipped.
    """
    phi = np.zeros(n_eq, dtype=np.float64)
    gamma = np.zeros(n_eq, dtype=np.float64)
    by_body: dict[BodyLike, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    cursor = 0
    for row in rows:
        phi_arr = np.atleast_1d(np.asarray(row.phi, dtype=np.float64))
        k = phi_arr.shape[0]
        gamma_arr = np.atleast_1d(np.asarray(row.gamma, dtype=np.float64))
        sl = slice(cursor, cursor + k)
        phi[sl] = phi_arr
        gamma[sl] = gamma_arr

        for body, (J_R, J_p) in row.blocks.items():
            if not has_dofs(body):
                continue
            if body not in by_body:
                by_body[body] = (
                    np.zeros((n_eq, 3), dtype=np.float64),
                    np.zeros((n_eq, 4), dtype=np.float64),
                )
            JR_buf, Jp_buf = by_body[body]
            JR_buf[sl] += np.atleast_2d(J_R).reshape(k, 3)
            Jp_buf[sl] += np.atleast_2d(J_p).reshape(k, 4)

        cursor += k

    if cursor != n_eq:
        raise AssertionError(f"row contributions span {cursor} eqs, expected {n_eq}")

    blocks: list[JacobianBlock] = [(b, JR, Jp) for b, (JR, Jp) in by_body.items()]
    return phi, blocks, gamma


# ─── Primitive constraint kernels ─────────────────────────────────────


def coincidence_rows(mi: Marker, mj: Marker) -> RowContribs:
    r"""Three rows enforcing :math:`\mathbf r_i^P - \mathbf r_j^P = \mathbf 0`.

    The spherical-joint kernel.
    """
    bi, bj = mi.parent, mj.parent
    u_i = bi.rotation_matrix @ mi.position_local
    u_j = bj.rotation_matrix @ mj.position_local
    phi = (bi.position + u_i) - (bj.position + u_j)

    blocks: dict[BodyLike, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    _add_block(blocks, bi, np.eye(3), dA_dp(u_i, bi.quaternion))
    _add_block(blocks, bj, -np.eye(3), -dA_dp(u_j, bj.quaternion))

    gamma = -(centripetal(bi.omega_global, u_i) - centripetal(bj.omega_global, u_j))
    return RowContribs(phi=phi, gamma=gamma, blocks=blocks)


def dot1_row(
    ma: Marker,
    va_body: ArrayLike,
    mb: Marker,
    vb_body: ArrayLike,
) -> RowContribs:
    r"""One row enforcing :math:`(A_a\,\mathbf v'_a)^\mathsf{T}(A_b\,\mathbf v'_b)=0`.

    Parameters
    ----------
    ma, mb
        Markers identifying the two parent bodies.
    va_body, vb_body
        Body-frame vectors on ``ma.parent`` and ``mb.parent`` respectively.
    """
    ba, bb = ma.parent, mb.parent
    u_a = ba.rotation_matrix @ np.asarray(va_body, dtype=np.float64)
    u_b = bb.rotation_matrix @ np.asarray(vb_body, dtype=np.float64)
    phi = float(u_a @ u_b)

    cross_ab = np.cross(u_a, u_b)
    Jp_a = cross_ab @ (2.0 * _q.E(ba.quaternion))
    Jp_b = -cross_ab @ (2.0 * _q.E(bb.quaternion))

    blocks: dict[BodyLike, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    _add_block(blocks, ba, np.zeros(3), Jp_a)
    _add_block(blocks, bb, np.zeros(3), Jp_b)

    gamma = gamma_dot1(ba.omega_global, u_a, bb.omega_global, u_b)
    return RowContribs(phi=phi, gamma=gamma, blocks=blocks)


def dot2_row(
    m_axis: Marker,
    v_axis_body: ArrayLike,
    mi_pt: Marker,
    mj_pt: Marker,
) -> RowContribs:
    r"""One row enforcing :math:`(A_a\,\mathbf v'_a)^\mathsf{T}\,\mathbf d=0`.

    with :math:`\mathbf d = \mathbf r_j^P - \mathbf r_i^P`, the vector
    pointing from marker ``i`` to marker ``j``. The "axis" vector
    :math:`\mathbf v'_a` is body-fixed on ``m_axis.parent``.

    Notes
    -----
    Decomposing the variation:

    .. math::

        \delta\Phi = (\mathbf u_a\!\times\!\mathbf d)\cdot\delta\boldsymbol\omega_a
        + \mathbf u_a\cdot\delta\mathbf R_j - \mathbf u_a\cdot\delta\mathbf R_i
        + (\mathbf u_a\!\times\!\mathbf u_i^{pt})\cdot\delta\boldsymbol\omega_i
        - (\mathbf u_a\!\times\!\mathbf u_j^{pt})\cdot\delta\boldsymbol\omega_j.

    Same-body contributions (e.g. ``m_axis.parent`` equal to ``mi_pt.parent``
    in a prismatic joint) are accumulated by :func:`assemble_rows`.
    """
    ba = m_axis.parent
    bi, bj = mi_pt.parent, mj_pt.parent
    u_a = ba.rotation_matrix @ np.asarray(v_axis_body, dtype=np.float64)
    u_i_pt = bi.rotation_matrix @ mi_pt.position_local
    u_j_pt = bj.rotation_matrix @ mj_pt.position_local
    d = (bj.position + u_j_pt) - (bi.position + u_i_pt)
    phi = float(u_a @ d)

    # Cache 2 E(p) per distinct body.
    twoE: dict[BodyLike, NDArray[np.float64]] = {}
    for b in (ba, bi, bj):
        if b not in twoE:
            twoE[b] = 2.0 * _q.E(b.quaternion)

    blocks: dict[BodyLike, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    # δu_a contribution to body a.
    _add_block(blocks, ba, np.zeros(3), np.cross(u_a, d) @ twoE[ba])
    # δd contributions to bodies i, j.
    _add_block(blocks, bi, -u_a, np.cross(u_a, u_i_pt) @ twoE[bi])
    _add_block(blocks, bj, +u_a, -np.cross(u_a, u_j_pt) @ twoE[bj])

    # γ for dot-2:
    # Φ̈ = ü_a·d + 2 u̇_a·ḋ + u_a·d̈,
    # q̈-independent part: centripetal(ω_a,u_a)·d + 2(ω_a×u_a)·ḋ
    #                     + u_a·[centripetal(ω_j,u_j^pt) - centripetal(ω_i,u_i^pt)]
    w_a, w_i, w_j = ba.omega_global, bi.omega_global, bj.omega_global
    v_i_mk = bi.velocity + np.cross(w_i, u_i_pt)
    v_j_mk = bj.velocity + np.cross(w_j, u_j_pt)
    d_dot = v_j_mk - v_i_mk
    gamma = -float(
        centripetal(w_a, u_a) @ d
        + 2.0 * (np.cross(w_a, u_a) @ d_dot)
        + u_a @ (centripetal(w_j, u_j_pt) - centripetal(w_i, u_i_pt))
    )
    return RowContribs(phi=phi, gamma=gamma, blocks=blocks)


# ─── Module-local helpers ─────────────────────────────────────────────


def _add_block(
    blocks: dict[BodyLike, tuple[NDArray[np.float64], NDArray[np.float64]]],
    body: BodyLike,
    J_R: NDArray[np.float64],
    J_p: NDArray[np.float64],
) -> None:
    """Accumulate ``(J_R, J_p)`` into ``blocks[body]`` (sum if already present)."""
    if body in blocks:
        old_R, old_p = blocks[body]
        blocks[body] = (old_R + J_R, old_p + J_p)
    else:
        blocks[body] = (J_R, J_p)
