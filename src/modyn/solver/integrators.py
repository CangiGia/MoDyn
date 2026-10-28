r"""Time integrators for the augmented index-3 multibody DAE.

The default scheme is the **generalized-:math:`\alpha`** method for constrained
mechanical systems (Arnold & Brüls, *Multibody Syst. Dyn.* 18, 2007;
Brüls, Cardona & Arnold, *Mech. Mach. Theory* 48, 2012). It integrates the
index-3 system

.. math::

    \mathbf M\,\ddot{\mathbf q} + \boldsymbol\Phi_q^\mathsf{T}\boldsymbol\lambda
        = \mathbf Q(t, \mathbf q, \dot{\mathbf q}), \qquad
    \boldsymbol\Phi(\mathbf q) = \mathbf 0

*directly*, enforcing the position-level constraints at every step through a
Newton iteration — no Baumgarte term and no post-step projection. Because the
constraints are satisfied at position level each step, the drift is bounded at
machine precision.

The Newton iteration matrix is the same saddle-point operator assembled by
:mod:`modyn.dynamics.eom`,

.. math::

    \mathbf S = \begin{bmatrix} \mathbf M & \boldsymbol\Phi_q^\mathsf{T} \\
    \boldsymbol\Phi_q & \mathbf 0 \end{bmatrix},

so the per-step work reuses the existing equation-of-motion assembly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

    # evaluate(t, q, qdot) -> (M, Phi_q, rhs_force, phi)
    # rhs_force = Q_e + Q_v; phi = constraint residual at trial state.
    EvalFn = Callable[
        [float, NDArray[np.float64], NDArray[np.float64]],
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ]

__all__ = ["GeneralizedAlpha", "StepResult"]


class StepResult(NamedTuple):
    r"""Outcome of a single generalized-:math:`\alpha` step.

    Attributes
    ----------
    q, qdot, qddot
        Updated generalized coordinates, velocities, and accelerations at
        :math:`t_{n+1}`, each shape ``(7n,)``.
    a
        Updated auxiliary acceleration variable (internal gen-:math:`\alpha`
        quantity), shape ``(7n,)``.
    lam
        Lagrange multipliers at :math:`t_{n+1}`, shape ``(m,)``.
    iterations
        Number of Newton iterations performed this step.
    residual_norm
        Final residual norm :math:`\max(\lVert r_1 \rVert, \lVert\Phi\rVert)`
        after convergence (or the last iteration value when not converged).
    converged
        ``True`` if the residual dropped below the requested tolerance.
    """

    q: NDArray[np.float64]
    qdot: NDArray[np.float64]
    qddot: NDArray[np.float64]
    a: NDArray[np.float64]
    lam: NDArray[np.float64]
    iterations: int
    residual_norm: float
    converged: bool


class GeneralizedAlpha:
    r"""Generalized-:math:`\alpha` integrator for the index-3 multibody DAE.

    Parameters
    ----------
    rho_inf
        Spectral radius at infinite frequency,
        :math:`\rho_\infty \in [0, 1]`. Controls high-frequency numerical
        dissipation: ``1.0`` gives a non-dissipative (energy-preserving in the
        linear regime) scheme; ``0.0`` annihilates the highest frequency mode
        in a single step. Default ``0.8``.

    Attributes
    ----------
    alpha_m, alpha_f, beta, gamma
        The four classical generalized-:math:`\alpha` coefficients derived
        from :attr:`rho_inf` via the optimal second-order unconditionally
        stable formulas of Chung & Hulbert (1993).
    """

    def __init__(self, rho_inf: float = 0.8) -> None:
        if not (0.0 <= rho_inf <= 1.0):
            raise ValueError(f"rho_inf must lie in [0, 1], got {rho_inf!r}.")
        self.rho_inf: float = float(rho_inf)
        self.alpha_m: float = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        self.alpha_f: float = rho_inf / (rho_inf + 1.0)
        self.gamma: float = 0.5 + self.alpha_f - self.alpha_m
        self.beta: float = 0.25 * (1.0 + self.alpha_f - self.alpha_m) ** 2

    def step(
        self,
        evaluate: EvalFn,
        t_n: float,
        h: float,
        q_n: NDArray[np.float64],
        qdot_n: NDArray[np.float64],
        qddot_n: NDArray[np.float64],
        a_n: NDArray[np.float64],
        lam_n: NDArray[np.float64],
        *,
        tol: float = 1e-8,
        max_iter: int = 20,
    ) -> StepResult:
        r"""Advance the state by one step of size ``h``.

        The step solves the nonlinear residual

        .. math::

            \begin{bmatrix} r_1 \\ r_2 \end{bmatrix} =
            \begin{bmatrix}
                \mathbf M\ddot{\mathbf q} + \boldsymbol\Phi_q^\mathsf{T}
                    \boldsymbol\lambda - \mathbf Q \\
                \boldsymbol\Phi / \beta'
            \end{bmatrix} = \mathbf 0

        by Newton iteration with the frozen saddle matrix
        :math:`\mathbf S = [[\mathbf M, \boldsymbol\Phi_q^\mathsf{T}],
        [\boldsymbol\Phi_q, \mathbf 0]]`. The scaling
        :math:`\beta' = \beta h^2 (1-\alpha_f)/(1-\alpha_m)` makes the
        off-diagonal blocks consistent (Brüls et al. 2012, eq. 19).

        Parameters
        ----------
        evaluate
            Callable ``evaluate(t, q, qdot) -> (M, Phi_q, rhs_force, phi)``.
            ``rhs_force`` is :math:`\mathbf Q_e + \mathbf Q_v`; ``phi`` is
            :math:`\boldsymbol\Phi(\mathbf q)`.
        t_n
            Current time :math:`t_n`; the step targets :math:`t_{n+1} = t_n + h`.
        h
            Step size (fixed).
        q_n, qdot_n, qddot_n
            State at :math:`t_n`.
        a_n
            Auxiliary acceleration at :math:`t_n`. Initialize to
            :math:`\ddot{\mathbf q}_0` at the start of a simulation.
        lam_n
            Lagrange multipliers at :math:`t_n`, used as the Newton predictor
            for :math:`\boldsymbol\lambda_{n+1}`.
        tol
            Convergence tolerance on the residual norm.
        max_iter
            Maximum number of Newton iterations per step.

        Returns
        -------
        StepResult
        """
        alpha_m = self.alpha_m
        alpha_f = self.alpha_f
        beta = self.beta
        gamma = self.gamma
        t_np1 = t_n + h
        n = q_n.shape[0]
        beta_prime = beta * h * h * (1.0 - alpha_f) / (1.0 - alpha_m)

        # Newton predictor: constant acceleration
        qddot = qddot_n.copy()
        lam = lam_n.copy()
        a_np1 = a_n.copy()
        q = q_n.copy()
        qdot = qdot_n.copy()

        res_norm = np.inf
        converged = False
        iterations = 0

        for _ in range(max_iter):
            iterations += 1
            # Auxiliary acceleration and Newmark update
            a_np1 = ((1.0 - alpha_f) * qddot + alpha_f * qddot_n - alpha_m * a_n) / (1.0 - alpha_m)
            q = q_n + h * qdot_n + h * h * (0.5 - beta) * a_n + h * h * beta * a_np1
            qdot = qdot_n + h * (1.0 - gamma) * a_n + h * gamma * a_np1

            M, Phi_q, rhs_force, phi = evaluate(t_np1, q, qdot)

            r1 = M @ qddot + Phi_q.T @ lam - rhs_force
            res_norm = max(float(np.linalg.norm(r1)), float(np.linalg.norm(phi)))
            if res_norm < tol:
                converged = True
                break

            m = phi.shape[0]
            St = np.zeros((n + m, n + m), dtype=np.float64)
            St[:n, :n] = M
            St[:n, n:] = Phi_q.T
            St[n:, :n] = Phi_q

            rhs = np.empty(n + m, dtype=np.float64)
            rhs[:n] = -r1
            rhs[n:] = -phi / beta_prime

            # Least squares keeps the Newton step defined for redundant constraints.
            delta, *_ = np.linalg.lstsq(St, rhs, rcond=None)
            qddot = qddot + delta[:n]
            lam = lam + delta[n:]

        return StepResult(
            q=q,
            qdot=qdot,
            qddot=qddot,
            a=a_np1,
            lam=lam,
            iterations=iterations,
            residual_norm=res_norm,
            converged=converged,
        )
