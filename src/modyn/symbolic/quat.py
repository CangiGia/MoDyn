r"""Euler-parameter algebra on CasADi ``SX`` expressions.

Same definitions as :mod:`modyn.math.quaternions` (Shabana convention):
:math:`A(\mathbf p) = (2e_0^2-1)\mathbf I + 2(\mathbf e\mathbf e^\mathsf{T} + e_0\tilde{\mathbf e})`,
:math:`G = [-\mathbf e,\ e_0\mathbf I - \tilde{\mathbf e}]`,
:math:`E = [-\mathbf e,\ e_0\mathbf I + \tilde{\mathbf e}]`.
"""

from __future__ import annotations

import casadi as ca

__all__ = ["A_sx", "E_sx", "G_sx", "skew_sx"]


def skew_sx(v):
    """Skew-symmetric matrix of a 3-vector."""
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0),
    )


def A_sx(p):
    """Rotation matrix ``A(p)`` (body to global) for ``p = [e0, e1, e2, e3]``."""
    e0 = p[0]
    e = p[1:4]
    return (2 * e0 * e0 - 1) * ca.SX.eye(3) + 2 * (ca.mtimes(e, e.T) + e0 * skew_sx(e))


def G_sx(p):
    """3x4 matrix ``G(p)`` such that ``omega' = 2 G(p) pdot``."""
    e0 = p[0]
    e = p[1:4]
    return ca.horzcat(-e, e0 * ca.SX.eye(3) - skew_sx(e))


def E_sx(p):
    """3x4 matrix ``E(p)`` such that ``omega = 2 E(p) pdot``."""
    e0 = p[0]
    e = p[1:4]
    return ca.horzcat(-e, e0 * ca.SX.eye(3) + skew_sx(e))
