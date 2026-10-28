"""Run an example of the PMD package and export its results in the PMD text format.

Used to produce the PMD reference of models for which no PMD export is
available (currently the double compound pendulum)::

    python run_pmd.py dcp [--t-final T] [--dt DT]

Positions, velocities and accelerations are written in SI units under
``<model folder>/PMD/<quantity>/PMD_<key>_cm_<quantity>.txt``. Joint
reactions are not exported: PMD stores the raw Lagrange multipliers, which
the thesis scripts convert to joint forces separately.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
from _common import SPECS, write_table

PMD_EXAMPLES = Path(__file__).resolve().parents[2] / "PMD" / "examples" / "scripting"
PMD_MODULES = {"dcp": "double_compound_pendulum.py"}


def run(key: str, t_final: float | None, dt: float | None) -> None:
    spec = SPECS[key]
    path = PMD_EXAMPLES / PMD_MODULES[key]
    module_spec = importlib.util.spec_from_file_location(f"pmd_example_{key}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    t_final = spec.t_final if t_final is None else t_final
    dt = spec.dt if dt is None else dt
    n = int(round(t_final / dt)) + 1
    clock = time.perf_counter()
    T, _ = module.model.solve(t_final=t_final, t_eval=np.linspace(0.0, t_final, n), ic_correct=True)
    print(f"PMD {spec.name}: {n - 1} steps, {time.perf_counter() - clock:.1f} s")

    folder = spec.path / "PMD"
    bodies = module.model.Bodies
    for body_spec in spec.bodies:
        body = bodies[body_spec.index]
        name = body.name
        getters = {
            "coordinates": (body.get_positions(), ("x", "y")),
            "velocities": (body.get_velocities(), ("dx", "dy")),
            "accelerations": (body.get_accelerations(), ("ddx", "ddy")),
        }
        for quantity, (values, comps) in getters.items():
            data = np.column_stack([values[c] for c in comps])
            write_table(
                folder / quantity / f"PMD_{body_spec.key}_cm_{quantity}.txt",
                T,
                [f"{name} / {c}" for c in comps],
                data,
            )
    print(f"  written to {folder}")
    print("  note: joint reactions are not exported (PMD stores raw multipliers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(PMD_MODULES))
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    args = parser.parse_args()
    run(args.model, args.t_final, args.dt)
