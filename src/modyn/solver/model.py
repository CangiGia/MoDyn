"""``SpatialMultibodyModel`` — top-level façade.

Owns the lists of bodies, joints, and force objects; triggers initial
assembly; and dispatches to the analysis drivers in this package.

Counterpart to PMD's ``PlanarMultibodyModel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..dynamics.coordinates import SystemCoordinates
from ..forces import Force, to_force_fn
from .dynamics import DynamicsResult, integrate_dynamics
from .kinematics import AssemblyResult, assemble_position, solve_velocity
from .statics import StaticsResult, find_equilibrium

if TYPE_CHECKING:
    from ..constraints.base import Constraint
    from ..model.body import RigidBody

__all__ = ["SpatialMultibodyModel"]


class SpatialMultibodyModel:
    """Top-level container for a 3-D rigid multibody system.

    Collects bodies, joints, and forces; provides convenience methods that
    call the low-level analysis drivers.

    Examples
    --------
    >>> model = SpatialMultibodyModel()
    >>> model.add_body(body)
    >>> model.add_joint(joint)
    >>> model.add_force(Gravity())
    >>> result = model.run_dynamics((0.0, 1.0), dt=1e-3)
    """

    def __init__(self) -> None:
        self._bodies: list[RigidBody] = []
        self._joints: list[Constraint] = []
        self._forces: list[Force] = []
        self._layout: SystemCoordinates | None = None

    # ── Registration ──────────────────────────────────────────────────

    def add_body(self, body: RigidBody) -> None:
        """Add a body to the model and invalidate the cached layout."""
        self._bodies.append(body)
        self._layout = None

    def add_joint(self, joint: Constraint) -> None:
        """Add a joint constraint to the model."""
        self._joints.append(joint)

    def add_force(self, force: Force) -> None:
        """Add a force element to the model."""
        self._forces.append(force)

    # ── Layout ────────────────────────────────────────────────────────

    @property
    def layout(self) -> SystemCoordinates:
        """Coordinate layout (built lazily from the registered bodies)."""
        if self._layout is None:
            self._layout = SystemCoordinates(self._bodies)
        return self._layout

    @property
    def bodies(self) -> list[RigidBody]:
        """Registered rigid bodies (ordered as added)."""
        return list(self._bodies)

    @property
    def joints(self) -> list[Constraint]:
        """Registered joint constraints."""
        return list(self._joints)

    @property
    def forces(self) -> list[Force]:
        """Registered force elements."""
        return list(self._forces)

    # ── Analysis drivers ──────────────────────────────────────────────

    def assemble(
        self,
        *,
        tol: float = 1e-10,
        max_iter: int = 50,
    ) -> AssemblyResult:
        """Project positions onto the constraint manifold and correct velocities.

        Calls :func:`~modyn.solver.kinematics.assemble_position` then
        :func:`~modyn.solver.kinematics.solve_velocity`.

        Returns
        -------
        AssemblyResult
            Position-assembly outcome (convergence, iterations, residual).
        """
        result = assemble_position(self.layout, self._joints, tol=tol, max_iter=max_iter)
        solve_velocity(self.layout, self._joints)
        return result

    def find_equilibrium(
        self,
        *,
        tol: float = 1e-8,
        max_iter: int = 50,
        reg: float = 1e-6,
    ) -> StaticsResult:
        """Find a static-equilibrium configuration.

        Wraps :func:`~modyn.solver.statics.find_equilibrium` using the
        registered forces.

        Returns
        -------
        StaticsResult
        """
        Qe_fn = to_force_fn(self._forces, self.layout) if self._forces else None
        return find_equilibrium(
            self.layout,
            self._joints,
            applied_forces=Qe_fn,
            tol=tol,
            max_iter=max_iter,
            reg=reg,
        )

    def run_dynamics(
        self,
        t_span: tuple[float, float],
        dt: float,
        *,
        rho_inf: float = 0.8,
        newton_tol: float = 1e-8,
        newton_max_iter: int = 20,
        project_initial: bool = True,
        assembly_tol: float = 1e-10,
        assembly_max_iter: int = 50,
    ) -> DynamicsResult:
        r"""Integrate the system dynamics.

        Wraps :func:`~modyn.solver.dynamics.integrate_dynamics` using the
        registered forces.

        Parameters
        ----------
        t_span
            ``(t0, tf)`` — start and end times.
        dt
            Fixed step size.
        rho_inf
            Spectral radius at infinite frequency for the
            generalised-:math:`\alpha` integrator.
        newton_tol
            Per-step Newton tolerance.
        newton_max_iter
            Maximum Newton iterations per step.
        project_initial
            Enforce position and velocity consistency before integration.
        assembly_tol, assembly_max_iter
            Forwarded to :func:`~modyn.solver.kinematics.assemble_position`
            when ``project_initial=True``.

        Returns
        -------
        DynamicsResult
        """
        Qe_fn = to_force_fn(self._forces, self.layout) if self._forces else None
        return integrate_dynamics(
            self.layout,
            self._joints,
            t_span,
            dt,
            applied_forces=Qe_fn,
            rho_inf=rho_inf,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
            project_initial=project_initial,
            assembly_tol=assembly_tol,
            assembly_max_iter=assembly_max_iter,
        )
