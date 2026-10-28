r"""Symbolic multibody system in quasi-velocities.

Builds, with CasADi ``SX``, the quantities of the formulation of
:mod:`modyn.dynamics.quasi`:

- the position constraints :math:`\boldsymbol\Phi(\mathbf q, t)`,
- the Jacobian :math:`\mathbf D = \boldsymbol\Phi_q\mathbf T` and the explicit
  time derivative :math:`\boldsymbol\Phi_t`, both by automatic differentiation,
- the velocity residual :math:`\boldsymbol\nu = \mathbf D\mathbf v + \boldsymbol\Phi_t`
  and the acceleration term :math:`\mathbf c` in
  :math:`\mathbf D\dot{\mathbf v} + \mathbf c = \mathbf 0`,
- the force vector :math:`\mathbf h(\mathbf q, \mathbf v, t)` (applied
  forces and gyroscopic terms),
- the semi-explicit index-3 DAE used by the collocation integrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from ..dynamics import quasi as _quasi
from .quat import G_sx
from .state import SymbolicState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..constraints.base import Constraint
    from ..dynamics.coordinates import SystemCoordinates
    from ..forces.base import Force

__all__ = ["SymbolicSystem"]


class SymbolicSystem:
    r"""CasADi description of constraints, forces and EoM of a model.

    Parameters
    ----------
    layout
        Coordinate layout.
    constraints
        Joint and driver constraints (all must implement ``phi_sym``).
    forces
        Force elements (all must implement ``wrench_sym``).

    Attributes
    ----------
    q, v, t
        ``SX`` symbols for positions, quasi-velocities and time.
    n_coords, n_vel, n_eq
        Sizes of :math:`\mathbf q`, :math:`\mathbf v` and :math:`\boldsymbol\Phi`.
    M
        Constant mass matrix, ``(6n, 6n)``.
    """

    def __init__(
        self,
        layout: SystemCoordinates,
        constraints: Sequence[Constraint],
        forces: Sequence[Force] = (),
    ) -> None:
        self.layout = layout
        self.constraints = list(constraints)
        self.forces = list(forces)
        nb = layout.n_bodies
        self.n_bodies = nb
        self.n_coords = 7 * nb
        self.n_vel = 6 * nb

        self.q = ca.SX.sym("q", self.n_coords)
        self.v = ca.SX.sym("v", self.n_vel)
        self.t = ca.SX.sym("t")
        state = SymbolicState(layout, self.q, self.v)
        self.state = state

        blocks = [c.phi_sym(state, self.t) for c in self.constraints]
        self.n_eq = sum(c.n_eq for c in self.constraints)
        self.Phi = ca.vertcat(*blocks) if blocks else ca.SX.zeros(0, 1)

        self.T = self._kinematic_map()
        Jq = ca.jacobian(self.Phi, self.q)
        self.D = ca.mtimes(Jq, self.T)
        self.Phi_t = ca.jacobian(self.Phi, self.t)
        self.nu = ca.mtimes(self.D, self.v) + self.Phi_t
        qdot = ca.mtimes(self.T, self.v)
        self.c = ca.mtimes(ca.jacobian(self.nu, self.q), qdot) + ca.jacobian(self.nu, self.t)
        self.h = self._force_vector()

        self.M = _quasi.mass_matrix_quasi(layout)
        self.Minv = np.linalg.inv(self.M)

        args_qt = [self.q, self.t]
        args_qvt = [self.q, self.v, self.t]
        self._f_phi = ca.Function("phi", args_qt, [self.Phi])
        self._f_D = ca.Function("D", args_qt, [self.D])
        self._f_phit = ca.Function("phi_t", args_qt, [self.Phi_t])
        self._f_c = ca.Function("c", args_qvt, [self.c])
        self._f_h = ca.Function("h", args_qvt, [self.h])

    # ── Construction helpers ──────────────────────────────────────────

    def _kinematic_map(self):
        T = ca.SX.zeros(self.n_coords, self.n_vel)
        for k, body in enumerate(self.layout.bodies):
            s = 7 * k
            w = 6 * k
            T[s : s + 3, w : w + 3] = ca.SX.eye(3)
            T[s + 3 : s + 7, w + 3 : w + 6] = 0.5 * G_sx(self.state.p(body)).T
        return T

    def _force_vector(self):
        h = ca.SX.zeros(self.n_vel, 1)
        for k, body in enumerate(self.layout.bodies):
            w = self.state.omega_body(body)
            J = ca.DM(body.inertia)
            h[6 * k + 3 : 6 * k + 6] = -ca.cross(w, ca.mtimes(J, w))
        for force in self.forces:
            for body, F, tau in force.wrench_sym(self.state, self.t):
                k = self.layout.offset(body) // 7
                h[6 * k : 6 * k + 3] = h[6 * k : 6 * k + 3] + F
                h[6 * k + 3 : 6 * k + 6] = h[6 * k + 3 : 6 * k + 6] + tau
        return h

    # ── Numeric evaluation ────────────────────────────────────────────

    def phi(self, q, t: float = 0.0) -> np.ndarray:
        """Constraint residual :math:`\\Phi(q, t)`, shape ``(m,)``."""
        return np.asarray(self._f_phi(q, t)).reshape(-1)

    def jacobian(self, q, t: float = 0.0) -> np.ndarray:
        """Jacobian :math:`D(q, t) = \\Phi_q T`, shape ``(m, 6n)``."""
        return np.asarray(self._f_D(q, t)).reshape(self.n_eq, self.n_vel)

    def phi_t(self, q, t: float = 0.0) -> np.ndarray:
        """Explicit time derivative :math:`\\Phi_t`, shape ``(m,)``."""
        return np.asarray(self._f_phit(q, t)).reshape(-1)

    def acceleration_term(self, q, v, t: float = 0.0) -> np.ndarray:
        """Term ``c`` of :math:`D\\dot v + c = 0`, shape ``(m,)``."""
        return np.asarray(self._f_c(q, v, t)).reshape(-1)

    def forces_vector(self, q, v, t: float = 0.0) -> np.ndarray:
        """Force vector :math:`h(q, v, t)` on the quasi-velocities, shape ``(6n,)``."""
        return np.asarray(self._f_h(q, v, t)).reshape(-1)

    def solve_accelerations(self, q, v, t: float = 0.0, rows=None):
        r"""Quasi-accelerations and multipliers at a given state.

        Solves :math:`[\mathbf M, \mathbf D^\mathsf{T}; \mathbf D, \mathbf 0]
        [\dot{\mathbf v}; \boldsymbol\lambda] = [\mathbf h; -\mathbf c]` on the
        selected constraint rows (all rows by default). With redundant rows
        the minimum-norm multipliers are returned.
        """
        D = self.jacobian(q, t)
        c = self.acceleration_term(q, v, t)
        if rows is not None:
            D, c = D[rows], c[rows]
        h = self.forces_vector(q, v, t)
        nv, m = self.n_vel, D.shape[0]
        if m == 0:
            return self.Minv @ h, np.zeros(0)
        S = np.zeros((nv + m, nv + m))
        S[:nv, :nv] = self.M
        S[:nv, nv:] = D.T
        S[nv:, :nv] = D
        sol, *_ = np.linalg.lstsq(S, np.concatenate((h, -c)), rcond=None)
        return sol[:nv], sol[nv:]

    # ── DAE for the collocation integrator ────────────────────────────

    def dae(self, rows=None) -> dict:
        r"""Semi-explicit index-3 DAE ``{'x', 'z', 'ode', 'alg'}``.

        Differential states :math:`\mathbf x = [\mathbf q, \mathbf v, \tau]`
        (``tau`` is the absolute time), algebraic states
        :math:`\mathbf z = [\boldsymbol\lambda, \boldsymbol\mu]`:

        .. math::

            \dot{\mathbf q} = \mathbf T\mathbf v + \mathbf P\boldsymbol\mu,\quad
            \mathbf M\dot{\mathbf v} = \mathbf h - \mathbf D^\mathsf{T}\boldsymbol\lambda,\quad
            \dot\tau = 1,\quad
            \mathbf 0 = [\boldsymbol\Phi;\ \mathbf p_k^\mathsf{T}\mathbf p_k - 1].

        :math:`\boldsymbol\mu` is the multiplier of the Euler-parameter
        normalization (:math:`\mathbf P` places :math:`\mathbf p_k` in the
        block of body :math:`k`); it vanishes on the exact solution.

        Parameters
        ----------
        rows
            Indices of the independent constraint rows to keep (all by default).
        """
        idx = list(range(self.n_eq)) if rows is None else [int(r) for r in rows]
        Phi = self.Phi[idx] if idx else ca.SX.zeros(0, 1)
        D = self.D[idx, :] if idx else ca.SX.zeros(0, self.n_vel)

        lam = ca.SX.sym("lam", len(idx))
        mu = ca.SX.sym("mu", self.n_bodies)

        P = ca.SX.zeros(self.n_coords, self.n_bodies)
        norm = []
        for k, body in enumerate(self.layout.bodies):
            p = self.state.p(body)
            P[7 * k + 3 : 7 * k + 7, k] = p
            norm.append(ca.mtimes(p.T, p) - 1)

        qdot = ca.mtimes(self.T, self.v) + ca.mtimes(P, mu)
        vdot = ca.mtimes(ca.DM(self.Minv), self.h - ca.mtimes(D.T, lam))
        return {
            "x": ca.vertcat(self.q, self.v, self.t),
            "z": ca.vertcat(lam, mu),
            "ode": ca.vertcat(qdot, vdot, ca.SX.ones(1)),
            "alg": ca.vertcat(Phi, *norm),
        }
