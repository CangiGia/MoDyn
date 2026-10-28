"""Run a validation model in MoDyn and export the results in the PMD text format.

Usage::

    python run_modyn.py fbl|msd|dcp [--t-final T] [--dt DT] [--method radau|generalized_alpha]

The files are written under ``<model folder>/MoDyn[_<method>]/`` using the
same names as the PMD exports, in SI units (m, m/s, m/s^2, N):
``MoDyn_<key>_cm_<quantity>.txt`` per body and ``MoDyn_<stem>.txt`` per joint
reaction (force exerted by the joint on the body of marker *i*, global frame,
the ADAMS convention).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from _common import SPECS, load_example, write_table


def run(key: str, t_final: float | None, dt: float | None, method: str = "radau", out: str | None = None):
    spec = SPECS[key]
    example = load_example(spec)
    model, parts = example.build_model()
    t_final = spec.t_final if t_final is None else t_final
    dt = spec.dt if dt is None else dt

    clock = time.perf_counter()
    result = model.solve((0.0, t_final), dt, method=method)
    elapsed = time.perf_counter() - clock
    if not result.success:
        raise RuntimeError("the integration did not complete successfully")

    folder = spec.path / (out or ("MoDyn" if method == "radau" else f"MoDyn_{method}"))
    quantities = {
        "coordinates": result.position,
        "velocities": result.velocity,
        "accelerations": result.acceleration,
    }
    suffix = {"coordinates": ("x", "y"), "velocities": ("dx", "dy"), "accelerations": ("ddx", "ddy")}
    for body in spec.bodies:
        name = parts["bodies"][body.index].name
        for quantity, getter in quantities.items():
            cols = [f"{name} / {c}" for c in suffix[quantity]]
            write_table(
                folder / quantity / f"MoDyn_{body.key}_cm_{quantity}.txt",
                result.times,
                cols,
                getter(body.index)[:, :2],
            )
    for reaction in spec.reactions:
        joint = parts["joints"][reaction.joint]
        force = model.reaction(result, joint, side="i", frame="global")
        write_table(
            folder / "reactions" / f"MoDyn_{reaction.stem}.txt",
            result.times,
            [f"{joint.name} / Fx", f"{joint.name} / Fy"],
            force[:, :2],
        )
    for label, stem in spec.scalars:
        if stem == "spring_force":
            spring = parts["spring"]
            # spring marker on the mass is offset from the centre of mass; ground marker at the origin
            length = result.position(0)[:, 1] + spring.marker_i.position_local[1]
            force = spring._k * (length - spring.natural_length) + spring._c * result.velocity(0)[:, 1]
            write_table(
                folder / "reactions" / f"MoDyn_{stem}.txt", result.times, [label], force[:, None]
            )

    print(f"{spec.name}: {method}, {result.times.size - 1} steps, {elapsed:.1f} s")
    print(f"  max constraint violation : {result.info.get('max_constraint_violation', float('nan')):.2e}")
    print(f"  max | |p| - 1 |          : {result.quaternion_norm_error():.2e}")
    if "dropped_rows" in result.info:
        print(f"  redundant rows removed   : {int(np.size(result.info['dropped_rows']))}")
    if key == "dcp":
        # Conservative system: the total energy must stay constant over the whole run.
        g = 9.80665
        kinetic = np.zeros(result.times.size)
        potential = np.zeros(result.times.size)
        for i, body in enumerate(parts["bodies"]):
            w = result.omega_body(i)
            kinetic += 0.5 * body.mass * np.sum(result.velocity(i) ** 2, axis=1)
            kinetic += 0.5 * np.einsum("ti,ij,tj->t", w, body.inertia, w)
            potential += body.mass * g * result.position(i)[:, 1]
        energy = kinetic + potential
        drift = np.max(np.abs(energy - energy[0]))
        print(f"  max |E - E0|             : {drift:.2e} J")
        print(f"  max |E - E0| / max(T)    : {drift / np.max(kinetic):.2e}")
        result.info["energy_drift"] = drift
    print(f"  written to {folder}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(SPECS))
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--method", default="radau", choices=["radau", "generalized_alpha"])
    args = parser.parse_args()
    run(args.model, args.t_final, args.dt, args.method)
