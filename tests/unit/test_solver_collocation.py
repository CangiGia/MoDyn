"""Tests for the Radau IIA collocation solver."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from scipy.integrate import solve_ivp  # noqa: E402

from modyn.builder import assemble_topology  # noqa: E402
from modyn.constraints import RevoluteJoint  # noqa: E402
from modyn.dynamics import SystemCoordinates  # noqa: E402
from modyn.forces import Gravity  # noqa: E402
from modyn.math import quaternions as _q  # noqa: E402
from modyn.model.body import RigidBody  # noqa: E402
from modyn.model.ground import Ground  # noqa: E402
from modyn.model.marker import Marker  # noqa: E402
from modyn.solver.collocation import integrate_radau  # noqa: E402


def _pendulum(theta0=0.3, a=0.5, mass=2.0, Jc=0.05):
    ground = Ground()
    body = RigidBody(mass=mass, inertia=np.diag([0.01, 0.02, Jc]))
    joint = RevoluteJoint(Marker(ground), Marker(body, position=(-a, 0.0, 0.0)), q0=[theta0])
    layout = SystemCoordinates([body])
    assemble_topology(layout, [joint])
    return layout, joint, body


def test_pendulum_matches_reference():
    g, a, mass, Jc = 9.81, 0.5, 2.0, 0.05
    layout, joint, body = _pendulum(0.3, a, mass, Jc)
    res = integrate_radau(layout, [joint], [Gravity((0.0, -g, 0.0))], (0.0, 2.0), 0.01)
    assert res.success

    def rhs(t, y):
        return [y[1], -mass * g * a * np.cos(y[0]) / (Jc + mass * a * a)]

    ref = solve_ivp(rhs, (0.0, 2.0), [0.3, 0.0], t_eval=res.times, rtol=1e-12, atol=1e-13)
    # body orientation angle about z from the Euler parameters
    p = res.quaternion(0)
    theta = 2.0 * np.arctan2(p[:, 3], p[:, 0])
    np.testing.assert_allclose(theta, ref.y[0], atol=1e-7)
    assert res.info["max_constraint_violation"] < 1e-9
    assert res.quaternion_norm_error() < 1e-9
    assert res.info["dropped_rows"].size == 0


def test_free_asymmetric_body_conserves_energy_and_momentum():
    J = np.diag([1.0, 2.0, 3.0])
    body = RigidBody(
        mass=1.5,
        inertia=J,
        quaternion=_q.from_axis_angle([1.0, 2.0, 0.5], 0.4),
        velocity=(0.2, -0.1, 0.3),
        omega_body=(1.0, 2.0, 0.5),
    )
    layout = SystemCoordinates([body])
    res = integrate_radau(layout, [], [], (0.0, 5.0), 0.01)
    assert res.success

    w = res.omega_body(0)
    kinetic_rot = 0.5 * np.einsum("ti,ij,tj->t", w, J, w)
    kinetic_tr = 0.5 * 1.5 * np.sum(res.velocity(0) ** 2, axis=1)
    energy = kinetic_rot + kinetic_tr
    assert np.max(np.abs(energy - energy[0])) / energy[0] < 1e-8

    L = np.array([_q.A(p) @ (J @ wi) for p, wi in zip(res.quaternion(0), w, strict=True)])
    assert np.max(np.linalg.norm(L - L[0], axis=1)) / np.linalg.norm(L[0]) < 1e-8
    assert res.quaternion_norm_error() < 1e-10
    # Euler's equations hold: J w' = -w x J w
    np.testing.assert_allclose(
        res.vdot[:, 3:], -np.cross(w, w @ J.T) @ np.linalg.inv(J).T, atol=1e-9
    )


def test_invalid_arguments():
    layout, joint, _ = _pendulum()
    with pytest.raises(ValueError, match="t_span"):
        integrate_radau(layout, [joint], [], (1.0, 0.5), 0.01)
    with pytest.raises(ValueError, match="dt"):
        integrate_radau(layout, [joint], [], (0.0, 1.0), -0.1)
    with pytest.raises(ValueError, match="divide"):
        integrate_radau(layout, [joint], [], (0.0, 1.0), 0.3)
    with pytest.raises(ValueError, match="rows"):
        integrate_radau(layout, [joint], [], (0.0, 1.0), 0.1, rows="bogus")
