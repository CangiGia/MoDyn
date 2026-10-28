r"""Abstract base class for all constraints.

Every concrete constraint exposes, evaluated at the current state stored on
the bodies it connects:

- :attr:`n_eq` : number of scalar equations contributed.
- :meth:`phi` : constraint residual :math:`\\Phi(q)\\in\\mathbb R^{n_{eq}}`,
  zero on the constraint manifold.
- :meth:`phi_q` : Jacobian blocks :math:`\\partial\\Phi/\\partial q_B` for
  each body ``B`` actually affected by the constraint. Each block is a pair
  ``(J_R, J_p)`` with ``J_R \\in \\mathbb R^{n_{eq}\\times 3}`` and
  ``J_p \\in \\mathbb R^{n_{eq}\\times 4}``.
- :meth:`gamma` : RHS of the acceleration-level equation,
  :math:`\\gamma = -\\dot\\Phi_q\\,\\dot q`, so that
  :math:`\\Phi_q\\,\\ddot q = \\gamma`.

The Euler-parameter normalization constraint ``p^T p - 1 = 0`` is **not**
the joint's responsibility; the dynamics layer appends it automatically
for every body.

References
----------
Shabana A. A., *Computational Dynamics*, 3rd ed., ch. 6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..core.base import Base

if TYPE_CHECKING:
    from ..model.body import RigidBody
    from ..model.ground import Ground

__all__ = ["Constraint", "JacobianBlock"]


#: One Jacobian block: ``(body, J_R, J_p)`` with shapes ``(n_eq, 3)`` and ``(n_eq, 4)``.
JacobianBlock = tuple["RigidBody | Ground", NDArray[np.float64], NDArray[np.float64]]


class Constraint(Base, ABC):
    """Abstract base for every holonomic constraint connecting bodies."""

    #: Number of scalar equations this constraint contributes.
    n_eq: int

    @abstractmethod
    def phi(self) -> NDArray[np.float64]:
        r"""Return the constraint residual :math:`\\Phi(q)`, shape ``(n_eq,)``."""

    @abstractmethod
    def phi_q(self) -> list[JacobianBlock]:
        r"""Return one Jacobian block per affected body.

        Returns
        -------
        list of (body, J_R, J_p)
            ``J_R`` is :math:`\\partial\\Phi/\\partial R_{body}` with shape
            ``(n_eq, 3)``, ``J_p`` is :math:`\\partial\\Phi/\\partial p_{body}`
            with shape ``(n_eq, 4)``. Ground bodies may be skipped from the
            list since their coordinates are fixed.
        """

    @abstractmethod
    def gamma(self) -> NDArray[np.float64]:
        r"""Return the acceleration-level RHS :math:`\\gamma`, shape ``(n_eq,)``."""

    def set_time(self, t: float) -> None:
        """Inject the current simulation time into the constraint.

        The default implementation is a no-op.  Rheonomic constraints
        (drivers) override this to update their internal time state before
        :meth:`phi`, :meth:`phi_q`, and :meth:`gamma` are called.
        """
