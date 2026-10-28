"""Unit tests for forces/springs.py, constraints/functions.py,
constraints/drivers.py, and constraints/primitive.py.

Test catalogue
--------------
test_tsda_zero_deformation
    TSDA at natural length → F=0, Q_e=0.
test_tsda_tension
    TSDA stretched by Δl → F=k*Δl, Q_R along unit vector.
test_tsda_compression
    TSDA compressed → negative F (compression).
test_tsda_damping
    Markers moving apart → positive l_dot → damping force in tension.
test_tsda_lazy_natural_length
    natural_length=None → set on first call to current distance.
test_tsda_actuator_callable
    Actuator callable contributes to force correctly.
test_tsda_ground_marker
    Marker on Ground: no KeyError, only one body in Q_e.

test_function_constant
    Constant(c): value=c, d=0, dd=0.
test_function_linear
    Linear(a,b): value, derivative, second_derivative correct.
test_function_sinusoidal
    Sinusoidal: value, derivative, second_derivative checked analytically.
test_function_polynomial
    Polynomial: degree-3 polynomial, derivatives via finite difference.
test_function_user_defined
    UserDefined with explicit callables.
test_function_callable_shorthand
    fn(t) is shorthand for fn.value(t).

test_coincidence_constraint
    CoincidenceConstraint: phi=0 when markers coincide.
test_dot1_constraint
    Dot1Constraint: phi=0 when vectors are orthogonal.
test_dot2_constraint
    Dot2Constraint: phi=0 when vectors are orthogonal to relative position.

test_revolute_driver_phi_at_zero_angle
    RevoluteDriver at θ=0: phi = u_xi·u_xj - cos(0) = 1-1 = 0.
test_revolute_driver_set_time
    set_time updates phi to reflect the new prescribed angle.
test_prismatic_driver_phi_at_zero
    PrismaticDriver at d=0 and markers coincident: phi=0.
test_prismatic_driver_set_time
    set_time(t) updates phi for a linear function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modyn.constraints import (
    CoincidenceConstraint,
    Constant,
    Dot1Constraint,
    Dot2Constraint,
    Linear,
    Polynomial,
    PrismaticDriver,
    RevoluteDriver,
    Sinusoidal,
    UserDefined,
)
from modyn.dynamics.coordinates import SystemCoordinates
from modyn.forces import TSDA
from modyn.model.body import RigidBody
from modyn.model.ground import Ground
from modyn.model.marker import Marker

pytestmark = pytest.mark.unit


# ─── helpers ──────────────────────────────────────────────────────────────────


def _body(*, R=(0, 0, 0), p=(1, 0, 0, 0), v=(0, 0, 0), mass=1.0):
    return RigidBody(
        mass=mass,
        inertia=np.eye(3),
        position=R,
        quaternion=p,
        velocity=v,
    )


# ─── TSDA ─────────────────────────────────────────────────────────────────────


def test_tsda_zero_deformation():
    """At natural length with no damping → F=0, Q_e=0."""
    b1 = _body(R=(0, 0, 0))
    b2 = _body(R=(1, 0, 0))
    layout = SystemCoordinates([b1, b2])
    m1 = Marker(b1, position=(0, 0, 0))
    m2 = Marker(b2, position=(0, 0, 0))
    tsda = TSDA(m1, m2, k=100.0, natural_length=1.0)
    Qe = tsda.generalized_force(layout, t=0.0)
    assert_allclose(Qe, 0.0, atol=1e-14)


def test_tsda_tension():
    """Stretched by Δl: Q_R on body 2 = k*Δl * e, on body 1 = -k*Δl * e."""
    k = 200.0
    l0 = 1.0
    dl = 0.5
    b1 = _body(R=(0, 0, 0))
    b2 = _body(R=(l0 + dl, 0, 0))
    layout = SystemCoordinates([b1, b2])
    m1 = Marker(b1, position=(0, 0, 0))
    m2 = Marker(b2, position=(0, 0, 0))
    tsda = TSDA(m1, m2, k=k, natural_length=l0)
    Qe = tsda.generalized_force(layout, t=0.0)
    F = k * dl
    assert_allclose(Qe[0:3], np.array([-F, 0, 0]), atol=1e-12)  # body 1 pulled +x
    assert_allclose(Qe[7:10], np.array([F, 0, 0]), atol=1e-12)  # body 2 pulled -x


def test_tsda_compression():
    """Compressed: force is negative (compression)."""
    b1 = _body(R=(0, 0, 0))
    b2 = _body(R=(0.5, 0, 0))  # shorter than l0=1
    layout = SystemCoordinates([b1, b2])
    m1 = Marker(b1, position=(0, 0, 0))
    m2 = Marker(b2, position=(0, 0, 0))
    tsda = TSDA(m1, m2, k=100.0, natural_length=1.0)
    Qe = tsda.generalized_force(layout, t=0.0)
    F = 100.0 * (0.5 - 1.0)  # = -50 N
    assert_allclose(Qe[0:3], np.array([-F, 0, 0]), atol=1e-12)  # -(-50) = +50 on body1


def test_tsda_damping():
    """Bodies moving apart: l_dot > 0 → damping contributes positive F_c."""
    c = 10.0
    vrel = 2.0
    b1 = _body(R=(0, 0, 0), v=(0, 0, 0))
    b2 = _body(R=(1, 0, 0), v=(vrel, 0, 0))
    layout = SystemCoordinates([b1, b2])
    m1 = Marker(b1, position=(0, 0, 0))
    m2 = Marker(b2, position=(0, 0, 0))
    tsda = TSDA(m1, m2, k=0.0, c=c, natural_length=1.0)
    Qe = tsda.generalized_force(layout, t=0.0)
    F = c * vrel
    assert_allclose(Qe[7:10], np.array([F, 0, 0]), atol=1e-12)


def test_tsda_lazy_natural_length():
    """natural_length=None → set lazily on first call."""
    b1 = _body(R=(0, 0, 0))
    b2 = _body(R=(3, 0, 0))
    layout = SystemCoordinates([b1, b2])
    m1 = Marker(b1, position=(0, 0, 0))
    m2 = Marker(b2, position=(0, 0, 0))
    tsda = TSDA(m1, m2, k=100.0)  # no natural_length
    assert tsda.natural_length is None
    Qe = tsda.generalized_force(layout, t=0.0)
    assert tsda.natural_length == pytest.approx(3.0)
    assert_allclose(Qe, 0.0, atol=1e-14)  # at natural length, F=0


def test_tsda_actuator_callable():
    """Actuator callable: F_act(t=1.0) = 7.0 → reflected in Q_e."""
    b1 = _body(R=(0, 0, 0))
    b2 = _body(R=(1, 0, 0))
    layout = SystemCoordinates([b1, b2])
    m1 = Marker(b1, position=(0, 0, 0))
    m2 = Marker(b2, position=(0, 0, 0))
    tsda = TSDA(m1, m2, k=0.0, natural_length=1.0, actuator=lambda t: 7.0 * t)
    Qe = tsda.generalized_force(layout, t=1.0)
    assert_allclose(Qe[7:10], np.array([7.0, 0, 0]), atol=1e-12)


def test_tsda_ground_marker():
    """Marker on Ground: no KeyError; only body DOFs are filled."""
    ground = Ground()
    body = _body(R=(1, 0, 0))
    layout = SystemCoordinates([body])
    mg = Marker(ground, position=(0, 0, 0))
    mb = Marker(body, position=(0, 0, 0))
    tsda = TSDA(mg, mb, k=100.0, natural_length=1.0)
    Qe = tsda.generalized_force(layout, t=0.0)
    assert Qe.shape == (7,)
    assert_allclose(Qe, 0.0, atol=1e-14)


# ─── Function types ───────────────────────────────────────────────────────────


def test_function_constant():
    f = Constant(3.14)
    assert f.value(0.0) == pytest.approx(3.14)
    assert f.derivative(0.0) == pytest.approx(0.0)
    assert f.second_derivative(0.0) == pytest.approx(0.0)
    assert f(1.5) == pytest.approx(3.14)


def test_function_linear():
    f = Linear(slope=2.0, intercept=-1.0)
    assert f.value(3.0) == pytest.approx(5.0)
    assert f.derivative(3.0) == pytest.approx(2.0)
    assert f.second_derivative(3.0) == pytest.approx(0.0)


def test_function_sinusoidal():
    A, omega, phi = 3.0, 2.0, 0.5
    f = Sinusoidal(amplitude=A, frequency=omega, phase=phi)
    t = 1.0
    assert f.value(t) == pytest.approx(A * math.sin(omega * t + phi))
    assert f.derivative(t) == pytest.approx(A * omega * math.cos(omega * t + phi))
    assert f.second_derivative(t) == pytest.approx(-A * omega**2 * math.sin(omega * t + phi))


def test_function_polynomial():
    # f(t) = 1 + 2t + 3t^2
    f = Polynomial([1.0, 2.0, 3.0])
    t = 2.0
    assert f.value(t) == pytest.approx(1 + 2 * t + 3 * t**2)
    # Finite-difference check of derivative
    h = 1e-6
    fd_d = (f.value(t + h) - f.value(t - h)) / (2 * h)
    assert f.derivative(t) == pytest.approx(fd_d, rel=1e-5)
    fd_dd = (f.value(t + h) - 2 * f.value(t) + f.value(t - h)) / h**2
    assert f.second_derivative(t) == pytest.approx(fd_dd, rel=1e-3)


def test_function_user_defined():
    f = UserDefined(f=lambda t: t**3, df=lambda t: 3 * t**2, ddf=lambda t: 6 * t)
    assert f.value(2.0) == pytest.approx(8.0)
    assert f.derivative(2.0) == pytest.approx(12.0)
    assert f.second_derivative(2.0) == pytest.approx(12.0)


def test_function_callable_shorthand():
    f = Linear(slope=1.0, intercept=0.0)
    assert f(3.0) == pytest.approx(f.value(3.0))


# ─── Primitive constraints ────────────────────────────────────────────────────


def test_coincidence_constraint():
    b = _body(R=(0, 0, 0))
    m = Marker(b, position=(0, 0, 0))
    mg = Marker(Ground(), position=(0, 0, 0))
    c = CoincidenceConstraint(mg, m)
    assert_allclose(c.phi(), 0.0, atol=1e-14)
    assert len(c.phi_q()) == 1  # only body b (Ground skipped)
    assert c.gamma().shape == (3,)


def test_dot1_constraint():
    """Two perpendicular body-fixed vectors at identity → phi=0."""
    b = _body()
    m = Marker(b)
    mg = Marker(Ground())
    # ex on body, ey on ground → ex · ey = 0
    c = Dot1Constraint(m, [1, 0, 0], mg, [0, 1, 0])
    assert c.phi() == pytest.approx(np.array([0.0]))


def test_dot2_constraint():
    """Axis on ground perpendicular to d=r_j - r_i → phi=0."""
    b = _body(R=(0, 1, 0))  # body at y=1
    mg = Marker(Ground(), position=(0, 0, 0))
    mb = Marker(b, position=(0, 0, 0))
    # d = (0,1,0); axis = (1,0,0) → dot = 0
    c = Dot2Constraint(Marker(Ground()), [1, 0, 0], mg, mb)
    assert c.phi() == pytest.approx(np.array([0.0]), abs=1e-14)


# ─── Revolute & Prismatic drivers ────────────────────────────────────────────


def test_revolute_driver_phi_at_zero_angle():
    """RevoluteDriver with f=Constant(0): u_xi·u_xj - cos(0) = 1 - 1 = 0."""
    ground = Ground()
    body = _body()
    mg = Marker(ground)  # identity orientation → x_body = [1,0,0]
    mb = Marker(body)
    fn = Constant(0.0)  # prescribe θ=0
    driver = RevoluteDriver(mg, mb, fn)
    assert_allclose(driver.phi(), np.array([0.0]), atol=1e-14)


def test_revolute_driver_set_time():
    """After set_time to π/2, cos(π/2)=0; phi = u_xi·u_xj - 0 = 1 (at identity)."""
    ground = Ground()
    body = _body()
    mg = Marker(ground)
    mb = Marker(body)
    driver = RevoluteDriver(mg, mb, Linear(slope=1.0))  # θ(t)=t
    driver.set_time(math.pi / 2)
    phi = driver.phi()
    # u_xi · u_xj = 1 (both body-fixed x at identity rotation), cos(π/2) = 0
    assert_allclose(phi, np.array([1.0]), atol=1e-14)


def test_prismatic_driver_phi_at_zero():
    """PrismaticDriver with f=Constant(0): u_zi · d = 0 when markers coincide."""
    ground = Ground()
    body = _body(R=(0, 0, 0))
    mg = Marker(ground, position=(0, 0, 0))
    mb = Marker(body, position=(0, 0, 0))
    fn = Constant(0.0)
    driver = PrismaticDriver(mg, mb, fn)
    assert_allclose(driver.phi(), np.array([0.0]), atol=1e-14)


def test_prismatic_driver_set_time():
    """PrismaticDriver with Linear(slope=1): at t=2, phi = d - 2 = 0 when d=2."""
    ground = Ground()
    body = _body(R=(0, 0, 2))  # z=2 → u_z·d = 2
    mg = Marker(ground, position=(0, 0, 0))
    mb = Marker(body, position=(0, 0, 0))
    fn = Linear(slope=1.0)  # f(t) = t
    driver = PrismaticDriver(mg, mb, fn)
    driver.set_time(2.0)
    # phi = u_z·d - f(2) = 2 - 2 = 0
    assert_allclose(driver.phi(), np.array([0.0]), atol=1e-14)
