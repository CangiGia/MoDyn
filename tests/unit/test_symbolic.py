"""Symbolic (CasADi, AD) layer versus the hand-written numeric constraints."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from modyn.constraints import (  # noqa: E402
    CoincidenceConstraint,
    CylindricalJoint,
    Dot1Constraint,
    Dot2Constraint,
    PlanarJoint,
    PrismaticDriver,
    PrismaticJoint,
    RevoluteDriver,
    RevoluteJoint,
    SphericalJoint,
    UniversalJoint,
)
from modyn.constraints.functions import Linear, Polynomial, Sinusoidal  # noqa: E402
from modyn.dynamics import SystemCoordinates  # noqa: E402
from modyn.dynamics import eom as _eom  # noqa: E402
from modyn.dynamics import quasi as _quasi  # noqa: E402
from modyn.forces import TSDA, Gravity, PointForce, Torque  # noqa: E402
from modyn.math import quaternions as _q  # noqa: E402
from modyn.model.body import RigidBody  # noqa: E402
from modyn.model.ground import Ground  # noqa: E402
from modyn.model.marker import Marker  # noqa: E402
from modyn.symbolic import SymbolicSystem  # noqa: E402


def _rand_unit(rng):
    return _q.normalize(rng.normal(size=4))


def _rand_marker(rng, parent):
    return Marker(
        parent,
        position=rng.normal(size=3),
        orientation=_q.A(_rand_unit(rng)),
    )


def _two_bodies(rng):
    J = np.diag(rng.uniform(0.2, 1.0, size=3))
    bodies = [
        RigidBody(
            mass=rng.uniform(0.5, 2.0),
            inertia=J,
            position=rng.normal(size=3),
            quaternion=_rand_unit(rng),
            velocity=rng.normal(size=3),
            omega_body=rng.normal(size=3),
        )
        for _ in range(2)
    ]
    return bodies


def _joint_factories(rng, b1, b2):
    g = Ground()
    mi, mj = _rand_marker(rng, b1), _rand_marker(rng, b2)
    mg = _rand_marker(rng, g)
    mi2 = _rand_marker(rng, b1)
    return {
        "spherical": lambda: SphericalJoint(mi, mj),
        "revolute": lambda: RevoluteJoint(mi, mj),
        "revolute_ground": lambda: RevoluteJoint(mg, mi),
        "prismatic": lambda: PrismaticJoint(mi, mj),
        "cylindrical": lambda: CylindricalJoint(mi, mj),
        "planar": lambda: PlanarJoint(mi, mj),
        "universal": lambda: UniversalJoint(mi, mj),
        "coincidence": lambda: CoincidenceConstraint(mi, mj),
        "dot1": lambda: Dot1Constraint(mi, [0.3, 0.5, 0.8], mj, [0.1, -0.7, 0.2]),
        "dot2": lambda: Dot2Constraint(mi2, [0.3, 0.5, 0.8], mi, mj),
        "revolute_driver": lambda: RevoluteDriver(mi, mj, Sinusoidal(0.7, 1.3, 0.2)),
        "revolute_driver_poly": lambda: RevoluteDriver(mg, mj, Polynomial([0.1, 0.5, 0.2])),
        "prismatic_driver": lambda: PrismaticDriver(mi, mj, Linear(0.4, 0.1)),
    }


NAMES = [
    "spherical",
    "revolute",
    "revolute_ground",
    "prismatic",
    "cylindrical",
    "planar",
    "universal",
    "coincidence",
    "dot1",
    "dot2",
    "revolute_driver",
    "revolute_driver_poly",
    "prismatic_driver",
]


@pytest.mark.parametrize("name", NAMES)
def test_symbolic_matches_numeric(name):
    rng = np.random.default_rng(42)
    b1, b2 = _two_bodies(rng)
    joint = _joint_factories(rng, b1, b2)[name]()
    t = 0.37
    joint.set_time(t)
    layout = SystemCoordinates([b1, b2])
    sym = SymbolicSystem(layout, [joint])

    q = layout.assemble_q()
    v = _quasi.assemble_velocity(layout)

    np.testing.assert_allclose(sym.phi(q, t), joint.phi(), atol=1e-12)
    D_num = _quasi.constraint_jacobian_quasi([joint], layout)
    np.testing.assert_allclose(sym.jacobian(q, t), D_num, atol=1e-10)
    gamma_v = _quasi.constraint_rhs_quasi([joint], layout)
    np.testing.assert_allclose(-sym.acceleration_term(q, v, t), gamma_v, atol=1e-9)


def test_phi_t_of_drivers():
    rng = np.random.default_rng(1)
    b1, b2 = _two_bodies(rng)
    mi, mj = _rand_marker(rng, b1), _rand_marker(rng, b2)
    layout = SystemCoordinates([b1, b2])
    sym = SymbolicSystem(layout, [RevoluteJoint(mi, mj), PrismaticDriver(mi, mj, Linear(0.4, 0.0))])
    q = layout.assemble_q()
    pt = sym.phi_t(q, 0.0)
    np.testing.assert_allclose(pt[:5], 0.0, atol=1e-14)
    np.testing.assert_allclose(pt[5], -0.4, atol=1e-12)


def test_symbolic_forces_match_numeric():
    rng = np.random.default_rng(7)
    b1, b2 = _two_bodies(rng)
    g = Ground()
    m1, m2 = _rand_marker(rng, b1), _rand_marker(rng, b2)
    mg = _rand_marker(rng, g)
    forces = [
        Gravity((0.0, -9.81, 0.0)),
        PointForce(m1, np.array([1.0, -2.0, 0.5])),
        Torque(b2, np.array([0.3, 0.1, -0.7])),
        TSDA(m1, m2, k=50.0, c=3.0, natural_length=0.4),
        TSDA(mg, m2, k=20.0, c=1.0, natural_length=0.2),
    ]
    layout = SystemCoordinates([b1, b2])
    sym = SymbolicSystem(layout, [], forces)

    Qe = sum(f.generalized_force(layout, 0.0) for f in forces)
    h_num = _quasi.applied_to_quasi(layout, Qe) + _quasi.gyroscopic_forces(layout)
    h_sym = sym.forces_vector(layout.assemble_q(), _quasi.assemble_velocity(layout), 0.0)
    np.testing.assert_allclose(h_sym, h_num, atol=1e-9)


def test_symbolic_accelerations_match_numeric_with_redundancy():
    rng = np.random.default_rng(3)
    ground = Ground()
    body = RigidBody(mass=1.0, inertia=np.eye(3), position=(1.0, 0.0, 0.0))
    m_g = Marker(ground, position=(0.0, 0.0, 0.0))
    m_b = Marker(body, position=(-1.0, 0.0, 0.0))
    joints = [RevoluteJoint(m_g, m_b)]
    layout = SystemCoordinates([body])
    force = Gravity((0.0, -9.81, 0.0))
    sym = SymbolicSystem(layout, joints, [force])
    vdot, lam = sym.solve_accelerations(layout.assemble_q(), np.zeros(6), 0.0)
    np.testing.assert_allclose(vdot[5], -9.81 / 2.0, atol=1e-9)
    # the numeric (hand-written) formulation gives the same reaction multipliers
    _, lam_num = _quasi.solve_accelerations_quasi(layout, joints, force.generalized_force(layout, 0.0))
    np.testing.assert_allclose(lam, lam_num, atol=1e-9)
    assert rng is not None


def test_euler_parameter_matrices_agree_with_numeric():
    import casadi as ca

    from modyn.symbolic import A_sx, E_sx, G_sx

    p = _q.normalize([0.3, -0.5, 0.7, 0.2])
    for f_sx, f_np in ((A_sx, _q.A), (E_sx, _q.E), (G_sx, _q.G)):
        np.testing.assert_allclose(np.asarray(ca.DM(ca.evalf(f_sx(ca.DM(p))))), f_np(p), atol=1e-14)


def test_eom_numeric_symbolic_normalization_rows_not_in_phi():
    rng = np.random.default_rng(5)
    b1, b2 = _two_bodies(rng)
    mi, mj = _rand_marker(rng, b1), _rand_marker(rng, b2)
    joint = SphericalJoint(mi, mj)
    layout = SystemCoordinates([b1, b2])
    sym = SymbolicSystem(layout, [joint])
    assert sym.n_eq == joint.n_eq
    assert _eom.n_constraint_eqs([joint], layout) == joint.n_eq + 2
