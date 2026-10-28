"""Unit tests for ``modyn.solver.statics`` and ``modyn.solver.model``.

Test catalogue
--------------
test_statics_no_force_on_manifold
    Zero force, already on manifold → equilibrium trivially (Q_e=0, Phi=0).
test_statics_gravity_constrained_body
    Body constrained to fixed position; gravity balanced by reaction.
test_statics_invalid_tol
    tol <= 0 raises ValueError.
test_statics_invalid_max_iter
    max_iter < 1 raises ValueError.
test_statics_result_attributes
    StaticsResult fields are all present.

test_model_add_and_layout
    add_body / add_joint / add_force populate lists; layout is lazy.
test_model_assemble
    model.assemble() runs position + velocity assembly.
test_model_run_dynamics
    model.run_dynamics() produces a DynamicsResult with correct shape.
test_model_find_equilibrium
    model.find_equilibrium() converges for a statically admissible system.
test_model_layout_invalidated_on_add_body
    Adding a body after first layout access invalidates the cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from modyn.constraints import SphericalJoint
from modyn.dynamics.coordinates import SystemCoordinates
from modyn.forces import Gravity
from modyn.model.body import RigidBody
from modyn.model.ground import Ground
from modyn.model.marker import Marker
from modyn.solver.dynamics import DynamicsResult
from modyn.solver.model import SpatialMultibodyModel
from modyn.solver.statics import StaticsResult, find_equilibrium

pytestmark = pytest.mark.unit


# ─── helpers ──────────────────────────────────────────────────────────────────


def _free_body(*, mass=1.0):
    body = RigidBody(
        mass=mass,
        inertia=np.eye(3),
        position=(0.0, 0.0, 0.0),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
    )
    layout = SystemCoordinates([body])
    return body, layout


# ─── statics: find_equilibrium ────────────────────────────────────────────────


def test_statics_no_force_on_manifold():
    """Zero force, body at origin → Q_e=0, Phi=0 → equilibrium on first check."""
    _, layout = _free_body()
    res = find_equilibrium(layout, [], applied_forces=None, tol=1e-8)
    assert isinstance(res, StaticsResult)
    assert res.converged
    assert res.equilibrium_residual < 1e-8
    assert res.constraint_residual < 1e-8


def test_statics_gravity_constrained_body():
    """Body spherically constrained to ground; gravity balanced by joint reaction."""
    ground = Ground()
    body = RigidBody(
        mass=1.0,
        inertia=np.eye(3),
        position=(0.0, 0.0, 0.0),
        quaternion=(1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
    )
    mg = Marker(ground, position=(0.0, 0.0, 0.0))
    mb = Marker(body, position=(0.0, 0.0, 0.0))
    joint = SphericalJoint(mg, mb)
    layout = SystemCoordinates([body])

    Qe = np.zeros(7, dtype=float)
    Qe[:3] = np.array([0.0, 0.0, -9.81])  # gravity on translational DOFs

    res = find_equilibrium(layout, [joint], applied_forces=Qe, tol=1e-8, max_iter=50)
    assert res.converged
    assert res.equilibrium_residual < 1e-8
    assert res.constraint_residual < 1e-8


def test_statics_invalid_tol():
    _, layout = _free_body()
    with pytest.raises(ValueError, match="tol"):
        find_equilibrium(layout, [], tol=0.0)


def test_statics_invalid_max_iter():
    _, layout = _free_body()
    with pytest.raises(ValueError, match="max_iter"):
        find_equilibrium(layout, [], max_iter=0)


def test_statics_result_attributes():
    _, layout = _free_body()
    res = find_equilibrium(layout, [])
    assert hasattr(res, "converged")
    assert hasattr(res, "iterations")
    assert hasattr(res, "equilibrium_residual")
    assert hasattr(res, "constraint_residual")
    assert hasattr(res, "lam")
    assert res.lam.shape == (1,)  # only normalization constraint


# ─── SpatialMultibodyModel ────────────────────────────────────────────────────


def test_model_add_and_layout():
    model = SpatialMultibodyModel()
    body = RigidBody(mass=1.0, inertia=np.eye(3))
    model.add_body(body)
    assert len(model.bodies) == 1
    assert model.layout.n_bodies == 1


def test_model_assemble():
    model = SpatialMultibodyModel()
    body = RigidBody(mass=1.0, inertia=np.eye(3))
    model.add_body(body)
    res = model.assemble()
    assert res.converged


def test_model_run_dynamics():
    model = SpatialMultibodyModel()
    body = RigidBody(mass=1.0, inertia=np.eye(3))
    model.add_body(body)
    model.add_force(Gravity(g_vec=(0.0, 0.0, -9.81)))

    result = model.run_dynamics((0.0, 0.01), dt=1e-3)
    assert isinstance(result, DynamicsResult)
    assert result.success
    nt = result.times.shape[0]
    assert result.q.shape == (nt, 7)


def test_model_find_equilibrium():
    """Body at fixed position (spherical joint) under gravity → equilibrium."""
    model = SpatialMultibodyModel()
    ground = Ground()
    body = RigidBody(
        mass=1.0,
        inertia=np.eye(3),
        position=(0.0, 0.0, 0.0),
        quaternion=(1.0, 0.0, 0.0, 0.0),
    )
    mg = Marker(ground, position=(0.0, 0.0, 0.0))
    mb = Marker(body, position=(0.0, 0.0, 0.0))
    joint = SphericalJoint(mg, mb)

    model.add_body(body)
    model.add_joint(joint)
    model.add_force(Gravity(g_vec=(0.0, 0.0, -9.81)))

    res = model.find_equilibrium(tol=1e-8)
    assert res.converged


def test_model_layout_invalidated_on_add_body():
    """Adding a body after first layout access should rebuild the layout."""
    model = SpatialMultibodyModel()
    b1 = RigidBody(mass=1.0, inertia=np.eye(3))
    model.add_body(b1)
    layout1 = model.layout  # access → cache
    b2 = RigidBody(mass=2.0, inertia=np.eye(3))
    model.add_body(b2)
    layout2 = model.layout  # should be rebuilt
    assert layout1 is not layout2
    assert layout2.n_bodies == 2
