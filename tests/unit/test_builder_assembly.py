"""Tests for ``modyn.builder.assembly`` (topology and hybrid assembly)."""

import numpy as np
import pytest

from modyn.builder import assemble_topology, build_topology, place_bodies
from modyn.constraints import (
    CylindricalJoint,
    PlanarJoint,
    PrismaticJoint,
    RevoluteJoint,
    SphericalJoint,
    UniversalJoint,
)
from modyn.dynamics import SystemCoordinates
from modyn.math import quaternions as _q
from modyn.model.body import RigidBody
from modyn.model.ground import Ground
from modyn.model.marker import Marker


def _body(**kw):
    return RigidBody(mass=1.0, inertia=np.eye(3), **kw)


def _rand_marker(rng, parent):
    q = _q.normalize(rng.normal(size=4))
    return Marker(parent, position=rng.normal(size=3), orientation=_q.A(q))


def test_chain_of_all_joint_types_is_placed_consistently():
    rng = np.random.default_rng(11)
    ground = Ground()
    bodies = [_body() for _ in range(5)]
    chain = [
        (SphericalJoint, rng.normal(size=3)),
        (RevoluteJoint, [0.7]),
        (PrismaticJoint, [0.35]),
        (CylindricalJoint, [-0.4, 0.2]),
        (PlanarJoint, [0.1, -0.3, 0.9]),
    ]
    joints = []
    prev = ground
    for body, (cls, q0) in zip(bodies, chain, strict=True):
        joints.append(cls(_rand_marker(rng, prev), _rand_marker(rng, body), q0=q0))
        prev = body

    layout = SystemCoordinates(bodies)
    topology = build_topology(bodies, joints)
    assert len(topology.tree) == 5 and not topology.cut_joints and topology.n_loops == 0
    place_bodies(topology)
    for j in joints:
        np.testing.assert_allclose(j.phi(), 0.0, atol=1e-12, err_msg=type(j).__name__)
    assert all(abs(np.linalg.norm(b.quaternion) - 1.0) < 1e-12 for b in layout.bodies)


def test_joint_placed_from_the_marker_j_side_of_the_parent():
    rng = np.random.default_rng(5)
    ground = Ground()
    b1, b2 = _body(), _body()
    # ground attached through marker_j, so the tree edge is traversed j -> i
    j1 = RevoluteJoint(_rand_marker(rng, b1), _rand_marker(rng, ground), q0=[0.4])
    j2 = PrismaticJoint(_rand_marker(rng, b1), _rand_marker(rng, b2), q0=[0.2])
    place_bodies(build_topology([b1, b2], [j1, j2]))
    np.testing.assert_allclose(j1.phi(), 0.0, atol=1e-12)
    np.testing.assert_allclose(j2.phi(), 0.0, atol=1e-12)


def test_revolute_coordinate_is_the_relative_angle():
    ground = Ground()
    body = _body()
    joint = RevoluteJoint(Marker(ground), Marker(body, position=(-1.0, 0.0, 0.0)), q0=[np.pi / 3])
    place_bodies(build_topology([body], [joint]))
    np.testing.assert_allclose(body.rotation_matrix[:2, 0], [np.cos(np.pi / 3), np.sin(np.pi / 3)])
    np.testing.assert_allclose(body.position, [np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0], atol=1e-14)


def _four_bar():
    L1, L2, L3, O4 = 0.04, 0.12, 0.08, 0.10
    th = np.deg2rad([40.0, 20.298, -122.605])
    ground = Ground()
    links = [_body(name=f"link_{k}") for k in (1, 2, 3)]
    l1, l2, l3 = links
    g1 = Marker(ground, position=(0, 0, 0))
    g3 = Marker(ground, position=(O4, 0, 0))
    l1_g, l1_l2 = Marker(l1, position=(-L1 / 2, 0, 0)), Marker(l1, position=(L1 / 2, 0, 0))
    l2_l1, l2_l3 = Marker(l2, position=(-L2 / 2, 0, 0)), Marker(l2, position=(L2 / 2, 0, 0))
    l3_l2, l3_g = Marker(l3, position=(-L3 / 2, 0, 0)), Marker(l3, position=(L3 / 2, 0, 0))
    joints = [
        RevoluteJoint(g1, l1_g, q0=[th[0]]),
        RevoluteJoint(l1_l2, l2_l1, q0=[th[1] - th[0]]),
        RevoluteJoint(l2_l3, l3_l2, q0=[th[2] - th[1]]),
        RevoluteJoint(l3_g, g3, q0=[th[2]]),
    ]
    return links, joints, (L1, L2, L3, O4, th)


def test_four_bar_topology_and_loop_closure():
    links, joints, (L1, L2, L3, O4, th) = _four_bar()
    layout = SystemCoordinates(links)
    topology, result = assemble_topology(layout, joints)
    assert len(topology.tree) == 3
    # BFS reaches crank and follower from the ground; the coupler-follower joint closes the loop
    assert topology.cut_joints == [joints[2]]
    assert {id(e.joint) for e in topology.tree} == {id(joints[0]), id(joints[1]), id(joints[3])}
    assert topology.n_loops == 1
    assert result.converged
    for j in joints:
        np.testing.assert_allclose(j.phi(), 0.0, atol=1e-9)
    # centroid of the crank from the first tree joint
    np.testing.assert_allclose(
        links[0].position[:2], 0.5 * L1 * np.array([np.cos(th[0]), np.sin(th[0])]), atol=1e-4
    )


def test_missing_q0_on_tree_joint_raises():
    ground = Ground()
    body = _body()
    joint = RevoluteJoint(Marker(ground), Marker(body))
    with pytest.raises(ValueError, match="q0"):
        place_bodies(build_topology([body], [joint]))


def test_disconnected_body_raises():
    ground = Ground()
    b1, b2 = _body(name="a"), _body(name="b")
    joint = RevoluteJoint(Marker(ground), Marker(b1), q0=[0.0])
    with pytest.raises(ValueError, match="not connected"):
        build_topology([b1, b2], [joint])


def test_q0_validation():
    ground = Ground()
    body = _body()
    with pytest.raises(ValueError, match="q0"):
        RevoluteJoint(Marker(ground), Marker(body), q0=[0.0, 1.0])
    with pytest.raises(NotImplementedError):
        UniversalJoint(Marker(ground), Marker(body)).relative_transform()
