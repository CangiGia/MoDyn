"""Analytical driver functions and their first two time derivatives.

Mirrors PMD's ``Function`` API so that example scripts port cleanly between
the two libraries.

Available concrete types
------------------------
Constant, Linear, Sinusoidal, Polynomial, UserDefined.

All subclasses implement
    ``value(t)``, ``derivative(t)``, ``second_derivative(t)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "Constant",
    "Function",
    "Linear",
    "Polynomial",
    "Sinusoidal",
    "UserDefined",
]


class Function(ABC):
    r"""Abstract scalar function of time with analytic derivatives.

    Subclasses must implement :meth:`value`, :meth:`derivative`, and
    :meth:`second_derivative`.  Drivers use the second derivative to
    assemble the acceleration-level right-hand side :math:`\gamma`.
    """

    @abstractmethod
    def value(self, t: float) -> float:
        """Return the function value at time ``t``."""

    @abstractmethod
    def derivative(self, t: float) -> float:
        """Return the first time derivative at ``t``."""

    @abstractmethod
    def second_derivative(self, t: float) -> float:
        """Return the second time derivative at ``t``."""

    def __call__(self, t: float) -> float:
        """Shorthand for :meth:`value`."""
        return self.value(t)


class Constant(Function):
    """Constant function :math:`f(t) = c`.

    Parameters
    ----------
    c
        Constant value.
    """

    def __init__(self, c: float) -> None:
        self._c = float(c)

    def value(self, t: float) -> float:  # noqa: D102
        return self._c

    def derivative(self, t: float) -> float:  # noqa: D102
        return 0.0

    def second_derivative(self, t: float) -> float:  # noqa: D102
        return 0.0


class Linear(Function):
    r"""Linear ramp :math:`f(t) = a\,t + b`.

    Parameters
    ----------
    slope
        Rate of change :math:`a`.
    intercept
        Value at :math:`t = 0`, default ``0.0``.
    """

    def __init__(self, slope: float, intercept: float = 0.0) -> None:
        self._a = float(slope)
        self._b = float(intercept)

    def value(self, t: float) -> float:  # noqa: D102
        return self._a * t + self._b

    def derivative(self, t: float) -> float:  # noqa: D102
        return self._a

    def second_derivative(self, t: float) -> float:  # noqa: D102
        return 0.0


class Sinusoidal(Function):
    r"""Sinusoidal function :math:`f(t) = A\,\sin(\omega\,t + \phi) + c`.

    Parameters
    ----------
    amplitude
        Peak amplitude :math:`A`.
    frequency
        Angular frequency :math:`\omega` [rad/s].
    phase
        Phase offset :math:`\phi` [rad], default ``0.0``.
    offset
        Constant vertical offset :math:`c`, default ``0.0``.
    """

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
        offset: float = 0.0,
    ) -> None:
        self._A = float(amplitude)
        self._omega = float(frequency)
        self._phi = float(phase)
        self._c = float(offset)

    def value(self, t: float) -> float:  # noqa: D102
        return self._A * np.sin(self._omega * t + self._phi) + self._c

    def derivative(self, t: float) -> float:  # noqa: D102
        return self._A * self._omega * np.cos(self._omega * t + self._phi)

    def second_derivative(self, t: float) -> float:  # noqa: D102
        return -self._A * self._omega**2 * np.sin(self._omega * t + self._phi)


class Polynomial(Function):
    r"""Polynomial :math:`f(t) = \sum_{i=0}^{n} c_i\,t^i`.

    Parameters
    ----------
    coeffs
        Sequence of coefficients ``[c0, c1, c2, …]`` with ascending degree.
        ``coeffs[0]`` is the constant term.
    """

    def __init__(self, coeffs: list[float]) -> None:
        self._c = np.asarray(coeffs, dtype=np.float64)

    def value(self, t: float) -> float:  # noqa: D102
        return float(sum(c * t**i for i, c in enumerate(self._c)))

    def derivative(self, t: float) -> float:  # noqa: D102
        return float(sum(i * c * t ** (i - 1) for i, c in enumerate(self._c) if i > 0))

    def second_derivative(self, t: float) -> float:  # noqa: D102
        return float(sum(i * (i - 1) * c * t ** (i - 2) for i, c in enumerate(self._c) if i > 1))


class UserDefined(Function):
    """User-supplied function with explicit analytic derivatives.

    Parameters
    ----------
    f
        ``f(t) -> float`` — function value.
    df
        ``df(t) -> float`` — first derivative.
    ddf
        ``ddf(t) -> float`` — second derivative.
    """

    def __init__(self, f, df, ddf) -> None:
        self._f, self._df, self._ddf = f, df, ddf

    def value(self, t: float) -> float:  # noqa: D102
        return float(self._f(t))

    def derivative(self, t: float) -> float:  # noqa: D102
        return float(self._df(t))

    def second_derivative(self, t: float) -> float:  # noqa: D102
        return float(self._ddf(t))
