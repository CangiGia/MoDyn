r"""Hybrid assembly: topology in joint coordinates, loop closure in Cartesian ones.

Same procedure as PMD, extended to 3D:

1. The connectivity graph (vertices = bodies with the ground as vertex 0,
   edges = joints) is explored by breadth-first search from the ground. Each
   joint that reaches a new body is a *tree joint*; the others are *cut
   joints*, so that :math:`n_L = n_j - n_b` independent loops remain.
2. Starting from the ground, each body is placed from its parent through the
   closed-form map of the tree joint, driven by the user-given joint
   coordinates ``q0`` (angle, displacement, rotation vector, ...).
3. The configuration is projected onto the full set of constraints with a
   minimum-norm Newton iteration (loop closure, Euler-parameter
   normalization included), exactly as in the planar case.

Only joints exposing ``marker_i``/``marker_j`` take part in the topology;
drivers and primitive constraints act as additional constraints.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, NamedTuple

from ..math.rotations import matrix_to_quat
from ..model.ground import Ground

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..constraints.base import Constraint
    from ..dynamics.coordinates import SystemCoordinates
    from ..model.body import RigidBody
    from ..solver.kinematics import AssemblyResult

__all__ = ["TreeEdge", "Topology", "assemble_topology", "build_topology", "place_bodies"]


class TreeEdge(NamedTuple):
    """Tree joint connecting an already placed ``parent`` to its ``child``."""

    parent: RigidBody | Ground
    child: RigidBody
    joint: Constraint


class Topology(NamedTuple):
    """Result of the breadth-first search.

    Attributes
    ----------
    tree
        Tree joints, in discovery order (parents before children).
    cut_joints
        Joints that close a loop.
    n_loops
        Number of independent loops, :math:`n_j - n_b`.
    """

    tree: list[TreeEdge]
    cut_joints: list[Constraint]
    n_loops: int


def _is_topological(joint: Constraint) -> bool:
    return hasattr(joint, "marker_i") and hasattr(joint, "marker_j")


def build_topology(bodies: Sequence[RigidBody], joints: Sequence[Constraint]) -> Topology:
    """Classify the joints into tree joints and cut joints by BFS from the ground.

    Raises
    ------
    ValueError
        If some body cannot be reached from the ground through the joints.
    """
    edges = [j for j in joints if _is_topological(j)]
    ground = Ground()
    visited: set[int] = {id(ground)}
    queue: deque = deque([ground])
    tree: list[TreeEdge] = []
    tree_ids: set[int] = set()

    while queue:
        node = queue.popleft()
        for joint in edges:
            if id(joint) in tree_ids:
                continue
            bi, bj = joint.marker_i.parent, joint.marker_j.parent
            if bi is node and id(bj) not in visited:
                parent, child = bi, bj
            elif bj is node and id(bi) not in visited:
                parent, child = bj, bi
            else:
                continue
            tree.append(TreeEdge(parent, child, joint))
            tree_ids.add(id(joint))
            visited.add(id(child))
            queue.append(child)

    unreached = [b.name for b in bodies if id(b) not in visited]
    if unreached:
        raise ValueError(f"bodies not connected to the ground through joints: {unreached}")

    cut = [j for j in edges if id(j) not in tree_ids]
    return Topology(tree=tree, cut_joints=cut, n_loops=len(edges) - len(bodies))


def place_bodies(topology: Topology) -> None:
    """Set the pose of every body from the joint coordinates of the tree joints.

    The joint coordinates ``q0`` give the pose of the child's marker frame
    relative to the parent's marker frame (the same convention as PMD,
    independent of which marker is ``i`` or ``j``); the child body is then
    placed so that its own marker lands on that frame.

    Raises
    ------
    ValueError
        If a tree joint has no initial joint coordinates ``q0``.
    """
    for parent, child, joint in topology.tree:
        if joint.q0 is None:
            raise ValueError(
                f"{type(joint).__name__} {joint.name!r} is a tree joint and needs q0 "
                "(initial joint coordinates)"
            )
        R_rel, t_rel = joint.relative_transform()
        mi, mj = joint.marker_i, joint.marker_j

        # q0 always describes the child's marker frame relative to the parent's
        # marker frame (PMD convention), whichever marker order the joint uses.
        if mi.parent is parent:
            m_parent, m_child = mi, mj
        else:
            m_parent, m_child = mj, mi
        A_mp = parent.rotation_matrix @ m_parent.orientation_local
        r_mp = parent.point_global(m_parent.position_local)
        A_mc = A_mp @ R_rel
        r_mc = r_mp + A_mp @ t_rel

        A_child = A_mc @ m_child.orientation_local.T
        child.position = r_mc - A_child @ m_child.position_local
        child.quaternion = matrix_to_quat(A_child)


def assemble_topology(
    layout: SystemCoordinates,
    constraints: Sequence[Constraint],
    *,
    t0: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> tuple[Topology, AssemblyResult]:
    """Place the bodies from ``q0`` and close the loops.

    Parameters
    ----------
    layout
        Coordinate layout; its bodies are updated in place.
    constraints
        All constraints of the model (joints, drivers, primitives).
    t0
        Initial time, injected into rheonomic constraints before the loop
        closure so that drivers are satisfied at the start of the motion.
    tol, max_iter
        Newton settings of the loop closure.

    Returns
    -------
    topology, assembly
        The classification of the joints and the Newton outcome.
    """
    from ..solver.kinematics import assemble_position

    topology = build_topology(layout.bodies, list(constraints))
    place_bodies(topology)
    for c in constraints:
        c.set_time(t0)
    result = assemble_position(layout, list(constraints), tol=tol, max_iter=max_iter)
    return topology, result
