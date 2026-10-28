"""Mass-spring-damper — vertical slider with a translational spring-damper.

A single body constrained by a prismatic joint to move along the global y
axis, attached to the ground by a TSDA. No gravity, as in the ADAMS and PMD
models used for the validation (``validation/mass-spring-damper``).

Parameters (m, kg, N, s): m = 10000 kg, k = 1e6 N/m, c = 1e4 N s/m, natural
length 0.8 m. The block starts at y = 1.25 m, at rest; the spring marker on the
block lies 0.25 m below its centre of mass, so the spring is initially 1.0 m
long, i.e. stretched by 0.2 m.
"""

from __future__ import annotations

import numpy as np

from modyn.constraints import PrismaticJoint
from modyn.forces import TSDA
from modyn.math.rotations import rotx
from modyn.model import Ground, Marker, RigidBody
from modyn.solver import SpatialMultibodyModel

T_FINAL = 15.0
DT = 5e-3

MASS = 10000.0
STIFFNESS = 1.0e6
DAMPING = 1.0e4
NATURAL_LENGTH = 0.8
SPRING_OFFSET = -0.25  # spring marker below the centre of mass
Y0 = 1.25


def build_model() -> tuple[SpatialMultibodyModel, dict]:
    """Return the model and a dictionary with its bodies, joints and forces."""
    ground = Ground()
    # Marker z-axes along the global y direction: the prismatic joint slides along y.
    to_y = rotx(-np.pi / 2)
    mass = RigidBody(mass=MASS, inertia=np.eye(3) * 1e-3, name="mass")

    g_slide = Marker(ground, position=(0.0, 0.0, 0.0), orientation=to_y)
    m_slide = Marker(mass, position=(0.0, 0.0, 0.0), orientation=to_y)
    slider = PrismaticJoint(g_slide, m_slide, q0=[Y0], name="mass_ground_joint")

    g_spring = Marker(ground, position=(0.0, 0.0, 0.0))
    m_spring = Marker(mass, position=(0.0, SPRING_OFFSET, 0.0))
    spring = TSDA(
        m_spring, g_spring, k=STIFFNESS, c=DAMPING, natural_length=NATURAL_LENGTH
    )

    model = SpatialMultibodyModel()
    model.add_body(mass)
    model.add_joint(slider)
    model.add_force(spring)
    return model, {"bodies": [mass], "joints": [slider], "spring": spring}


if __name__ == "__main__":
    model, parts = build_model()
    result = model.solve((0.0, T_FINAL), DT)
    print(f"success: {result.success}")
    print(f"final position y = {result.position(0)[-1, 1]:.6f} m")
    print(f"max constraint violation: {result.info['max_constraint_violation']:.2e}")
