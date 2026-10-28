"""Common result container of the dynamic solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["SimulationResult"]


@dataclass
class SimulationResult:
    r"""Trajectory returned by every dynamic solver.

    Attributes
    ----------
    times
        Time instants, shape ``(nt,)``.
    q
        Positions :math:`[\mathbf R, \mathbf p]` per body, shape ``(nt, 7n)``.
    v
        Quasi-velocities :math:`[\dot{\mathbf R}, \boldsymbol\omega']` per body
        (angular velocity in the body frame), shape ``(nt, 6n)``.
    vdot
        Quasi-accelerations, shape ``(nt, 6n)``.
    lam
        Constraint multipliers on **all** constraint rows (minimum-norm when
        the constraints are redundant), shape ``(nt, m)``.
    method
        Name of the solver that produced the result.
    success
        ``True`` if the integration completed without convergence failures.
    info
        Solver-specific diagnostics (kept rows, runtime, iterations, ...).
    """

    times: NDArray[np.float64]
    q: NDArray[np.float64]
    v: NDArray[np.float64]
    vdot: NDArray[np.float64]
    lam: NDArray[np.float64]
    method: str
    success: bool = True
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_bodies(self) -> int:
        """Number of moving bodies."""
        return self.q.shape[1] // 7

    def position(self, body_index: int) -> NDArray[np.float64]:
        """Centroid positions of body ``body_index``, shape ``(nt, 3)``."""
        return self.q[:, 7 * body_index : 7 * body_index + 3]

    def quaternion(self, body_index: int) -> NDArray[np.float64]:
        """Euler parameters of body ``body_index``, shape ``(nt, 4)``."""
        return self.q[:, 7 * body_index + 3 : 7 * body_index + 7]

    def velocity(self, body_index: int) -> NDArray[np.float64]:
        """Centroid velocities of body ``body_index``, shape ``(nt, 3)``."""
        return self.v[:, 6 * body_index : 6 * body_index + 3]

    def omega_body(self, body_index: int) -> NDArray[np.float64]:
        """Body-frame angular velocities of body ``body_index``, shape ``(nt, 3)``."""
        return self.v[:, 6 * body_index + 3 : 6 * body_index + 6]

    def acceleration(self, body_index: int) -> NDArray[np.float64]:
        """Centroid accelerations of body ``body_index``, shape ``(nt, 3)``."""
        return self.vdot[:, 6 * body_index : 6 * body_index + 3]

    def quaternion_norm_error(self) -> float:
        """Maximum deviation of ``||p||`` from one over all bodies and instants."""
        p = self.q.reshape(self.q.shape[0], self.n_bodies, 7)[:, :, 3:]
        return float(np.max(np.abs(np.linalg.norm(p, axis=2) - 1.0)))
