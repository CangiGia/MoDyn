"""``SpatialMultibodyModel`` — top-level façade.

Owns the lists of bodies, joints, and force objects; triggers initial
assembly; and dispatches to the analysis drivers in this package.

Counterpart to PMD's ``PlanarMultibodyModel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..builder import Topology, assemble_topology
from ..dynamics import quasi as _quasi
from ..dynamics.coordinates import SystemCoordinates
from ..forces import Force, to_force_fn
from .dynamics import DynamicsResult, integrate_dynamics
from .kinematics import AssemblyResult, assemble_position, solve_velocity
from .result import SimulationResult
from .statics import StaticsResult, find_equilibrium

if TYPE_CHECKING:
    from ..constraints.base import Constraint
    from ..model.body import RigidBody
    from ..symbolic import SymbolicSystem

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
        self._symbolic: SymbolicSystem | None = None
        self._topology: Topology | None = None

    # ── Registration ──────────────────────────────────────────────────

    def add_body(self, body: RigidBody) -> None:
        """Add a body to the model and invalidate the cached layout."""
        self._bodies.append(body)
        self._layout = None
        self._symbolic = None

    def add_joint(self, joint: Constraint) -> None:
        """Add a joint constraint to the model."""
        self._joints.append(joint)
        self._symbolic = None

    def add_force(self, force: Force) -> None:
        """Add a force element to the model."""
        self._forces.append(force)
        self._symbolic = None

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

    @property
    def topology(self) -> Topology | None:
        """Tree/cut-joint classification of the last topological assembly, if any."""
        return self._topology

    def symbolic(self) -> SymbolicSystem:
        """CasADi description of the model (cached until the model changes)."""
        from ..symbolic import SymbolicSystem

        if self._symbolic is None:
            self._symbolic = SymbolicSystem(self.layout, self._joints, self._forces)
        return self._symbolic

    # ── Analysis drivers ──────────────────────────────────────────────

    def assemble(
        self,
        *,
        topology: bool | None = None,
        t0: float = 0.0,
        tol: float = 1e-10,
        max_iter: int = 50,
    ) -> AssemblyResult:
        """Bring the model to a consistent initial configuration.

        With ``topology=True`` (or ``None`` when some joint carries initial
        joint coordinates ``q0``) the bodies are placed from the joint
        coordinates along the BFS spanning tree and the loops are closed with
        Newton (the hybrid assembly of the formulation,
        :func:`~modyn.builder.assemble_topology`). Otherwise the user-given
        poses are projected onto the constraints with
        :func:`~modyn.solver.kinematics.assemble_position`. In both cases the
        velocities are then corrected with
        :func:`~modyn.solver.kinematics.solve_velocity`, including the
        explicit time dependence of the drivers at ``t0``.

        Returns
        -------
        AssemblyResult
            Position-assembly outcome (convergence, iterations, residual).
        """
        use_topology = (
            any(j.q0 is not None for j in self._joints) if topology is None else topology
        )
        if use_topology:
            self._topology, result = assemble_topology(
                self.layout, self._joints, t0=t0, tol=tol, max_iter=max_iter
            )
        else:
            for c in self._joints:
                c.set_time(t0)
            result = assemble_position(self.layout, self._joints, tol=tol, max_iter=max_iter)
        solve_velocity(self.layout, self._joints, t=t0)
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

    def solve(
        self,
        t_span: tuple[float, float],
        dt: float,
        *,
        method: str = "radau",
        assemble: bool = True,
        **options,
    ) -> SimulationResult:
        r"""Run a dynamic analysis with the chosen solver.

        Parameters
        ----------
        t_span, dt
            ``(t0, tf)`` and the (uniform) step of the output grid.
        method
            ``"radau"`` (default): direct index-3 integration with Radau IIA
            collocation in quasi-velocities, the solver of the hybrid
            formulation (:func:`~modyn.solver.collocation.integrate_radau`);
            ``"generalized_alpha"``: generalized-:math:`\alpha` on the
            Euler-parameter form (:func:`~modyn.solver.dynamics.integrate_dynamics`).
        assemble
            Run :meth:`assemble` at ``t0`` first (topological placement from
            ``q0`` when available, loop closure, consistent velocities).
        **options
            Forwarded to the selected solver.

        Returns
        -------
        SimulationResult
            Positions, quasi-velocities, quasi-accelerations and multipliers
            on all constraint rows, in a solver-independent format.
        """
        if assemble:
            self.assemble(t0=t_span[0])
        if method == "radau":
            from .collocation import integrate_radau

            return integrate_radau(
                self.layout, self._joints, self._forces, t_span, dt, **options
            )
        if method in ("generalized_alpha", "gen_alpha"):
            from .postprocess import simulation_from_dynamics

            options.setdefault("project_initial", False)
            raw = self.run_dynamics(t_span, dt, **options)
            return simulation_from_dynamics(self.layout, self._joints, self._forces, raw)
        raise ValueError(f"unknown method {method!r}; use 'radau' or 'generalized_alpha'")

    def reaction(
        self,
        result: SimulationResult,
        joint: Constraint,
        side: str = "i",
        frame: str = "global",
    ) -> np.ndarray:
        """Force and torque exerted by ``joint`` on the body of marker ``side``.

        See :func:`~modyn.solver.reactions.joint_reaction`.
        """
        from .reactions import joint_reaction

        return joint_reaction(self.symbolic(), result, joint, side=side, frame=frame)
