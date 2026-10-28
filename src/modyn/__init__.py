"""MoDyn — 3D rigid multibody dynamics (spatial evolution of PMD).

3D rigid multibody simulation with Euler-parameter orientation and
absolute-coordinate DAE formulation (Shabana, *Computational Dynamics*).

The public façade exposes the top-level model class plus the most commonly
used model entities. Subpackages can also be imported directly when finer
granularity is needed.

See ``ARCHITECTURE.md`` at the repository root for design decisions and
import-graph rules.
"""

__version__ = "0.0.1"

# Public façade — imports are kept lazy/minimal during scaffolding.
# Re-exports will be added module-by-module as implementations land.

__all__ = [
    "__version__",
]
