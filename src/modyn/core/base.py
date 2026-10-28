"""Base class with automatic per-subclass instance counting and naming.

Every model entity (bodies, markers, joints, forces, …) inherits from
:class:`Base`. The class keeps an independent instance counter **per
subclass** and exposes a stable, human-readable default name when the
user does not supply one.

This module imports nothing from the rest of MoDyn.
"""

from __future__ import annotations

from typing import ClassVar

__all__ = ["Base"]


class Base:
    """Common parent of all MoDyn model entities.

    Subclasses get:

    - per-class instance counting via ``Base.count(cls)``;
    - a stable identifier ``self.id`` (1-based, unique within the subclass);
    - an optional ``self.name``; falls back to ``f"{ClassName}#{id}"``.

    Notes
    -----
    Counters live on the leaf class — i.e. the class that actually defines
    a counter — so that creating a ``RigidBody`` does not increment the
    counter of an unrelated subclass of :class:`Base`.

    Reset all counters with :meth:`reset_all_counts` (used by tests).
    """

    # Per-leaf-class registry, mapping class -> running count.
    _counts: ClassVar[dict[type, int]] = {}

    def __init__(self, name: str | None = None) -> None:
        cls = type(self)
        n = Base._counts.get(cls, 0) + 1
        Base._counts[cls] = n
        self._id: int = n
        self._name: str | None = name

    # ── Identity ──────────────────────────────────────────────────────

    @property
    def id(self) -> int:
        """1-based unique identifier within this exact subclass."""
        return self._id

    @property
    def name(self) -> str:
        """Human-readable name; defaults to ``f"{ClassName}#{id}"``."""
        return self._name if self._name is not None else f"{type(self).__name__}#{self._id}"

    @name.setter
    def name(self, value: str | None) -> None:
        self._name = value

    # ── Class-level helpers ───────────────────────────────────────────

    @classmethod
    def count(cls) -> int:
        """Return the number of instances of ``cls`` created so far."""
        return Base._counts.get(cls, 0)

    @classmethod
    def reset_count(cls) -> None:
        """Reset the instance counter of this exact class."""
        Base._counts[cls] = 0

    @staticmethod
    def reset_all_counts() -> None:
        """Reset every per-class counter (used by test fixtures)."""
        Base._counts.clear()

    # ── Dunder ────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Return ``<ClassName id=N name='...'>``."""
        return f"<{type(self).__name__} id={self._id} name={self.name!r}>"
