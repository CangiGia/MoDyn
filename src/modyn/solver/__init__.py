"""Top-level solver façade and analysis drivers.

Modules
-------
model       : ``SpatialMultibodyModel`` — public façade wiring everything.
kinematics  : position / velocity / acceleration analyses.
statics     : equilibrium analysis.
dynamics    : DAE time integration.
integrators : adapters around scipy and (later) custom integrators.
"""

from .dynamics import DynamicsResult, integrate_dynamics
from .integrators import GeneralizedAlpha, StepResult
from .kinematics import AssemblyResult, assemble_position, solve_velocity
from .model import SpatialMultibodyModel
from .statics import StaticsResult, find_equilibrium

__all__ = [
    "AssemblyResult",
    "DynamicsResult",
    "GeneralizedAlpha",
    "SpatialMultibodyModel",
    "StaticsResult",
    "StepResult",
    "assemble_position",
    "find_equilibrium",
    "integrate_dynamics",
    "solve_velocity",
]
