"""Core foundations: Base counter, units, logging setup, generic utilities.

This package has **no internal MoDyn dependencies**. Anything imported from
here must be safe to use from every other subpackage.
"""

from .base import Base
from .logging import configure_logging, get_logger
from .units import GRAVITY_SI, SI, UnitSystem
from .utils import (
    as_float_array,
    ensure_finite,
    ensure_non_negative,
    ensure_positive,
    ensure_shape,
    ensure_symmetric_positive_definite,
)

__all__ = [
    "GRAVITY_SI",
    "SI",
    "Base",
    "UnitSystem",
    "as_float_array",
    "configure_logging",
    "ensure_finite",
    "ensure_non_negative",
    "ensure_positive",
    "ensure_shape",
    "ensure_symmetric_positive_definite",
    "get_logger",
]
