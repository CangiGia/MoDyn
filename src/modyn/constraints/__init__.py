"""Constraints: kinematic joints connecting bodies.

Every joint is a :class:`~modyn.constraints.base.Constraint` subclass
exposing :meth:`phi`, :meth:`phi_q` and :meth:`gamma` (Shabana eqs. 6.66
-- 6.74). The Euler-parameter normalization constraint
:math:`\\mathbf p^\\mathsf{T}\\mathbf p - 1 = 0` is appended automatically
by the dynamics layer for every body, so joints do not handle it.

Each joint lives in its own module named after it: see
:mod:`~modyn.constraints.spherical`, :mod:`~modyn.constraints.revolute`,
etc.
"""

from .base import Constraint, JacobianBlock
from .cylindrical import CylindricalJoint
from .drivers import PrismaticDriver, RevoluteDriver
from .functions import Constant, Function, Linear, Polynomial, Sinusoidal, UserDefined
from .planar import PlanarJoint
from .primitive import CoincidenceConstraint, Dot1Constraint, Dot2Constraint
from .prismatic import PrismaticJoint
from .revolute import RevoluteJoint
from .spherical import SphericalJoint
from .universal import UniversalJoint

__all__ = [
    "CoincidenceConstraint",
    "Constant",
    "Constraint",
    "CylindricalJoint",
    "Dot1Constraint",
    "Dot2Constraint",
    "Function",
    "JacobianBlock",
    "Linear",
    "PlanarJoint",
    "Polynomial",
    "PrismaticDriver",
    "PrismaticJoint",
    "RevoluteDriver",
    "RevoluteJoint",
    "Sinusoidal",
    "SphericalJoint",
    "UniversalJoint",
    "UserDefined",
]
