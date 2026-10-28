"""Tests for the model-level solve API, reactions and solver agreement."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from modyn.constraints import RevoluteJoint, SphericalJoint  # noqa: E402
from modyn.forces import Gravity  # noqa: E402
from modyn.math import quaternions as _q  # noqa: E402
from modyn.model import Ground, Marker, RigidBody  # noqa: E402
from modyn.solver import SpatialMultibodyModel  # noqa: E402


def _pendulum_model(theta0=0.4, a=0.5, mass=2.0, Jc=0.05):
    ground = Ground()
    body = RigidBody(mass=mass, inertia=np.diag([0.01, 0.02, Jc]), name="rod")
    joint = RevoluteJoint(Marker(ground), Marker(body, position=(-a, 0.0, 0.0)), q0=[theta0])
    model = SpatialMultibodyModel()
    model.add_body(body)
    model.add_joint(joint)
    model.add_force(Gravity((0.0, -9.81, 0.0)))
    return model, joint, body


def test_solvers_agree_on_the_pendulum():
    model, _, _ = _pendulum_model()
    r_radau = model.solve((0.0, 0.5), 1e-3)
    model2, _, _ = _pendulum_model()
    r_alpha = model2.solve((0.0, 0.5), 1e-3, method="generalized_alpha", rho_inf=1.0)
    assert r_radau.method == "radau" and r_alpha.method == "generalized_alpha"
    np.testing.assert_allclose(r_radau.position(0), r_alpha.position(0), atol=2e-5)
    np.testing.assert_allclose(r_radau.omega_body(0), r_alpha.omega_body(0), atol=2e-3)
    np.testing.assert_allclose(r_radau.lam, r_alpha.lam, atol=2e-2)


def test_unknown_method_raises():
    model, _, _ = _pendulum_model()
    with pytest.raises(ValueError, match="unknown method"):
        model.solve((0.0, 1.0), 0.1, method="euler")


def test_static_reaction_equals_weight():
    """A pendulum hanging at rest from a hinge: the hinge carries the weight."""
    model, joint, body = _pendulum_model(theta0=-np.pi / 2)  # CM straight below the hinge
    result = model.solve((0.0, 0.1), 0.01)
    on_body = model.reaction(result, joint, side="j")
    np.testing.assert_allclose(on_body[:, 0], 0.0, atol=1e-8)
    np.testing.assert_allclose(on_body[:, 1], 2.0 * 9.81, atol=1e-8)
    # equal and opposite on the ground side
    on_ground = model.reaction(result, joint, side="i")
    np.testing.assert_allclose(on_ground[:, :3], -on_body[:, :3], atol=1e-10)
    # the marker frame of the ground is the global one
    np.testing.assert_allclose(model.reaction(result, joint, "i", "marker"), on_ground, atol=1e-12)


def test_reaction_argument_validation():
    model, joint, _ = _pendulum_model()
    result = model.solve((0.0, 0.05), 0.01)
    with pytest.raises(ValueError, match="side"):
        model.reaction(result, joint, side="k")
    with pytest.raises(ValueError, match="frame"):
        model.reaction(result, joint, frame="local")


def test_assemble_uses_topology_when_q0_is_given():
    model, joint, body = _pendulum_model(theta0=0.7)
    model.assemble()
    assert model.topology is not None and len(model.topology.tree) == 1
    np.testing.assert_allclose(joint.phi(), 0.0, atol=1e-12)


def test_spherical_pendulum_conserves_energy_and_vertical_angular_momentum():
    g, mass = 9.81, 1.5
    J = np.diag([0.04, 0.06, 0.09])
    ground = Ground()
    body = RigidBody(
        mass=mass,
        inertia=J,
        position=(0.3, -0.4, 0.2),
        quaternion=_q.from_axis_angle([1.0, 0.5, -0.3], 0.6),
        velocity=(0.0, 0.0, 0.0),
        omega_body=(0.3, -0.2, 0.5),
    )
    s = np.array([-0.1, 0.2, -0.05])
    pivot = body.point_global(s)
    joint = SphericalJoint(Marker(ground, position=pivot), Marker(body, position=s))
    model = SpatialMultibodyModel()
    model.add_body(body)
    model.add_joint(joint)
    model.add_force(Gravity((0.0, -g, 0.0)))

    result = model.solve((0.0, 4.0), 0.01)
    assert result.success
    assert result.info["max_constraint_violation"] < 1e-9
    assert result.quaternion_norm_error() < 1e-9

    w = result.omega_body(0)
    v = result.velocity(0)
    R = result.position(0)
    energy = (
        0.5 * mass * np.sum(v**2, axis=1)
        + 0.5 * np.einsum("ti,ij,tj->t", w, J, w)
        + mass * g * R[:, 1]
    )
    assert np.max(np.abs(energy - energy[0])) < 1e-6 * max(1.0, abs(energy[0]))

    L = np.array(
        [
            mass * np.cross(Ri - pivot, vi) + _q.A(p) @ (J @ wi)
            for Ri, vi, wi, p in zip(R, v, w, result.quaternion(0), strict=True)
        ]
    )
    np.testing.assert_allclose(L[:, 1], L[0, 1], atol=1e-7)  # gravity along y
