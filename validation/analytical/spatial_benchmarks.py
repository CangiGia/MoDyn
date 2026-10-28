"""Spatial benchmarks with an independent reference solution.

The three models validated against ADAMS and PMD are planar. These two cases
exercise the full 3D rotational dynamics (Euler parameters, gyroscopic terms,
spherical joint under gravity) against reference solutions obtained from the
classical equations of motion integrated with a tight-tolerance explicit
scheme:

- **torque-free asymmetric rigid body** (Euler's equations, body-frame
  angular velocity, intermediate-axis spin);
- **heavy symmetric top** (Lagrange top: spherical joint, gravity,
  Euler-angle equations of motion).

Usage::

    python spatial_benchmarks.py [--no-show] [--no-figure] [--convergence]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import COLORS  # noqa: E402

from modyn.constraints import SphericalJoint  # noqa: E402
from modyn.forces import Gravity  # noqa: E402
from modyn.math import quaternions as _q  # noqa: E402
from modyn.math.rotations import matrix_to_quat, rotx, rotz  # noqa: E402
from modyn.model import Ground, Marker, RigidBody  # noqa: E402
from modyn.solver import SpatialMultibodyModel  # noqa: E402

HERE = Path(__file__).resolve().parent


def free_rigid_body(t_final: float = 20.0, dt: float = 0.01):
    """Torque-free asymmetric body spinning close to its intermediate axis."""
    J = np.diag([1.0, 2.0, 3.0])
    w0 = np.array([0.05, 2.0, 0.05])  # intermediate axis: unstable, large amplitude motion
    body = RigidBody(mass=1.0, inertia=J, omega_body=w0, name="body")
    model = SpatialMultibodyModel()
    model.add_body(body)
    result = model.solve((0.0, t_final), dt)

    def euler(_t, w):
        return -np.linalg.solve(J, np.cross(w, J @ w))

    ref = solve_ivp(
        euler, (0.0, t_final), w0, t_eval=result.times, method="DOP853", rtol=1e-13, atol=1e-14
    )
    w = result.omega_body(0)
    L = np.array([_q.A(p) @ (J @ wi) for p, wi in zip(result.quaternion(0), w, strict=True)])
    energy = 0.5 * np.einsum("ti,ij,tj->t", w, J, w)
    metrics = {
        "max |omega' - omega'_ref| [rad/s]": float(np.max(np.abs(w - ref.y.T))),
        "max |dL| / |L|": float(np.max(np.linalg.norm(L - L[0], axis=1)) / np.linalg.norm(L[0])),
        "max |dE| / E": float(np.max(np.abs(energy - energy[0])) / energy[0]),
        "max ||p|| - 1": result.quaternion_norm_error(),
    }
    return result, ref.y.T, metrics


def lagrange_top(t_final: float = 5.0, dt: float = 0.005):
    """Heavy symmetric top on a spherical joint, spin and precession about the vertical."""
    g, mass, ell = 9.81, 0.5, 0.1
    I1p, I3 = 8.0e-3, 3.0e-3  # inertia about the pivot: transverse and axial (I1p > m l^2)
    theta0, phi0 = np.deg2rad(40.0), np.deg2rad(10.0)
    omega3, phidot0 = 80.0, 1.0  # spin and initial precession rate

    # reference: Lagrange top, theta from the upward vertical, conserved p_psi, p_phi
    p_psi = I3 * omega3
    p_phi = I1p * phidot0 * np.sin(theta0) ** 2 + p_psi * np.cos(theta0)

    def rhs(_t, y):
        th, ph, thd = y
        phd = (p_phi - p_psi * np.cos(th)) / (I1p * np.sin(th) ** 2)
        thdd = (
            I1p * phd**2 * np.sin(th) * np.cos(th) - p_psi * phd * np.sin(th) + mass * g * ell * np.sin(th)
        ) / I1p
        return [thd, phd, thdd]

    times = np.arange(0.0, t_final + dt / 2, dt)
    ref = solve_ivp(
        rhs, (0.0, t_final), [theta0, phi0, 0.0], t_eval=times, method="DOP853", rtol=1e-13, atol=1e-14
    )
    axis = np.column_stack(
        (np.sin(ref.y[0]) * np.sin(ref.y[1]), -np.sin(ref.y[0]) * np.cos(ref.y[1]), np.cos(ref.y[0]))
    )
    cm_ref = ell * axis

    # MoDyn model: body frame with z along the symmetry axis, spherical joint at the origin
    R0 = rotz(phi0) @ rotx(theta0)
    Icm = np.diag([I1p - mass * ell**2, I1p - mass * ell**2, I3])
    w0_body = np.array([0.0, phidot0 * np.sin(theta0), omega3])
    r0 = ell * R0[:, 2]
    body = RigidBody(
        mass=mass,
        inertia=Icm,
        position=r0,
        quaternion=matrix_to_quat(R0),
        velocity=np.cross(R0 @ w0_body, r0),  # consistent with the fixed pivot
        omega_body=w0_body,
        name="top",
    )
    ground = Ground()
    joint = SphericalJoint(Marker(ground), Marker(body, position=(0.0, 0.0, -ell)))
    model = SpatialMultibodyModel()
    model.add_body(body)
    model.add_joint(joint)
    model.add_force(Gravity((0.0, 0.0, -g)))
    result = model.solve((0.0, t_final), dt)

    w = result.omega_body(0)
    v = result.velocity(0)
    energy = (
        0.5 * mass * np.sum(v**2, axis=1)
        + 0.5 * np.einsum("ti,ij,tj->t", w, Icm, w)
        + mass * g * result.position(0)[:, 2]
    )
    metrics = {
        "max |r_cm - r_cm_ref| [m]": float(np.max(np.linalg.norm(result.position(0) - cm_ref, axis=1))),
        "max |omega_3 - omega_3_ref| [rad/s]": float(np.max(np.abs(w[:, 2] - omega3))),
        "max |dE| / |E|": float(np.max(np.abs(energy - energy[0])) / abs(energy[0])),
        "max ||p|| - 1": result.quaternion_norm_error(),
        "max constraint violation": float(result.info["max_constraint_violation"]),
    }
    return result, cm_ref, metrics


def convergence(steps=(0.01, 0.005, 0.0025, 0.00125)) -> list[tuple[float, float, float]]:
    """Error of the Lagrange top against the reference for decreasing step sizes."""
    rows = []
    for dt in steps:
        _, _, m = lagrange_top(5.0, dt)
        rows.append((dt, m["max |r_cm - r_cm_ref| [m]"], m["max |dE| / |E|"]))
    return rows


def main(show: bool, figure: bool, with_convergence: bool = False) -> None:
    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from plot_style import setup_thesis_plots

    free, w_ref, m_free = free_rigid_body()
    top, cm_ref, m_top = lagrange_top()

    for title, metrics in (("Torque-free asymmetric rigid body", m_free), ("Lagrange top", m_top)):
        print(f"\n{title}")
        for name, value in metrics.items():
            print(f"  {name:<40s} {value:.3e}")

    if with_convergence:
        print("\nLagrange top: convergence with the step size")
        print(f"  {'dt [s]':>10s} {'max |dr_cm| [m]':>16s} {'ratio':>8s} {'max |dE|/|E|':>14s}")
        previous = None
        for dt, err, de in convergence():
            ratio = f"{previous / err:8.1f}" if previous else " " * 8
            print(f"  {dt:10.5f} {err:16.3e} {ratio} {de:14.3e}")
            previous = err

    if not figure:
        return
    setup_thesis_plots()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7})
    ref_style = {"color": COLORS["ADAMS"], "lw": 2.2}
    num_style = {"color": COLORS["MoDyn"], "lw": 1.2, "ls": ":"}

    fig, axes = plt.subplots(3, 1, figsize=(4.8, 5.4), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.plot(free.times, w_ref[:, i], **ref_style)
        ax.plot(free.times, free.omega_body(0)[:, i], **num_style)
        ax.set_ylabel(f"$\\omega'_{i + 1}$ [rad/s]")
    axes[-1].set_xlabel("Time [s]")
    handles = [
        plt.Line2D([0], [0], label="Reference", **ref_style),
        plt.Line2D([0], [0], label="MoDyn", **num_style),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig.savefig(HERE / "ANALYTICAL_free_rigid_body.pdf")

    fig2, axes2 = plt.subplots(3, 1, figsize=(4.8, 5.4), constrained_layout=True)
    for i, ax in enumerate(axes2):
        ax.plot(top.times, cm_ref[:, i], **ref_style)
        ax.plot(top.times, top.position(0)[:, i], **num_style)
        ax.set_ylabel(f"${'xyz'[i]}_{{cm}}$ [m]")
    axes2[-1].set_xlabel("Time [s]")
    fig2.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False)
    fig2.savefig(HERE / "ANALYTICAL_lagrange_top.pdf")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--convergence", action="store_true", help="step-size convergence table")
    args = parser.parse_args()
    main(show=not args.no_show, figure=not args.no_figure, with_convergence=args.convergence)
