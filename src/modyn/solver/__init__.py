"""Top-level solver façade and analysis drivers.

Modules
-------
model       : ``SpatialMultibodyModel`` — public façade wiring everything.
kinematics  : position / velocity analyses.
statics     : equilibrium analysis.
collocation : Radau IIA direct index-3 integration (CasADi), the default solver.
dynamics    : generalized-alpha time integration on the Euler-parameter form.
integrators : the generalized-alpha kernel.
result      : solver-independent ``SimulationResult``.
reactions   : joint reactions from the multipliers.
"""

from .collocation import integrate_radau
from .dynamics import DynamicsResult, integrate_dynamics
from .integrators import GeneralizedAlpha, StepResult
from .kinematics import AssemblyResult, assemble_position, solve_velocity
from .model import SpatialMultibodyModel
from .reactions import joint_reaction
from .result import SimulationResult
from .statics import StaticsResult, find_equilibrium

__all__ = [
    "AssemblyResult",
    "DynamicsResult",
    "GeneralizedAlpha",
    "SimulationResult",
    "SpatialMultibodyModel",
    "StaticsResult",
    "StepResult",
    "assemble_position",
    "find_equilibrium",
    "integrate_dynamics",
    "integrate_radau",
    "joint_reaction",
    "solve_velocity",
]
