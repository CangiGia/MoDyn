"""Coherent unit system and gravity constant.

MoDyn works in **coherent units**: all quantities entering the equations
of motion (mass, length, time, force, torque, …) must already be expressed
in a consistent system. The :class:`UnitSystem` describes the system in
use so that I/O layers can convert to/from external data.

Default is SI: kg / m / s.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GRAVITY_SI", "SI", "UnitSystem"]

#: Standard gravitational acceleration on Earth, m/s².
GRAVITY_SI: float = 9.80665


@dataclass(frozen=True, slots=True)
class UnitSystem:
    """Description of a coherent unit system.

    Attributes
    ----------
    length : str
        Length unit symbol (e.g. ``"m"``).
    mass : str
        Mass unit symbol (e.g. ``"kg"``).
    time : str
        Time unit symbol (e.g. ``"s"``).
    angle : str
        Angle unit symbol (``"rad"`` recommended for solver internals).
    gravity : float
        Magnitude of the gravitational acceleration in this system.
    """

    length: str = "m"
    mass: str = "kg"
    time: str = "s"
    angle: str = "rad"
    gravity: float = GRAVITY_SI

    @property
    def force(self) -> str:
        """Symbolic force unit derived from the base units."""
        return f"{self.mass}·{self.length}/{self.time}²"

    @property
    def torque(self) -> str:
        """Symbolic torque unit derived from the base units."""
        return f"{self.mass}·{self.length}²/{self.time}²"


SI: UnitSystem = UnitSystem()
