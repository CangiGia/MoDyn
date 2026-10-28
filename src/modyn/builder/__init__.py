"""Initial assembly utilities.

Topological placement of the bodies from joint coordinates (BFS spanning tree)
followed by a minimum-norm Newton loop closure on ``Φ(q) = 0`` (including
Euler-parameter normalization), as in PMD.
"""

from .assembly import Topology, TreeEdge, assemble_topology, build_topology, place_bodies

__all__ = ["Topology", "TreeEdge", "assemble_topology", "build_topology", "place_bodies"]
