"""Symbolic (CasADi) layer: constraints, forces and the EoM as expressions.

The numeric classes of :mod:`modyn.constraints` and :mod:`modyn.forces` keep
their hand-written Jacobians. This package rebuilds the same equations as
CasADi ``SX`` expressions so that the constraint Jacobian, the explicit time
derivatives and the second-order terms are obtained by automatic
differentiation, as done by PMD.

CasADi is imported only here (and lazily by the ``*_sym`` methods), so the
rest of the package stays importable without it.
"""

from .primitives import coincidence, dot1, dot2
from .quat import A_sx, E_sx, G_sx, skew_sx
from .state import SymbolicState
from .system import SymbolicSystem

__all__ = [
    "A_sx",
    "E_sx",
    "G_sx",
    "SymbolicState",
    "SymbolicSystem",
    "coincidence",
    "dot1",
    "dot2",
    "skew_sx",
]
