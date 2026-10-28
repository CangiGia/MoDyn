"""3D mathematical primitives.

This package is the foundation of the entire library. It MUST stay free of
any MoDyn-internal dependency (only NumPy/SciPy allowed) and MUST be the
most heavily unit-tested package — bugs here propagate everywhere.

Modules
-------
vectors      : skew/tilde matrices, cross-product helpers.
rotations    : rotation matrices, Euler/Cardan angle conversions.
quaternions  : Euler-parameter algebra, ``A(p)``, ``G(p)``, ``E(p)``,
               relations between ``ṗ`` and ``ω``.
transforms   : SE(3) homogeneous transforms.
"""

from . import quaternions, rotations, transforms, vectors
from .quaternions import A, E, G
from .transforms import Transform
from .vectors import skew, unskew

__all__ = [
    "A",
    "E",
    "G",
    "Transform",
    "quaternions",
    "rotations",
    "skew",
    "transforms",
    "unskew",
    "vectors",
]
