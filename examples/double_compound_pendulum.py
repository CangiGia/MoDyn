"""Double compound pendulum — two links in series, planar motion in 3D.

Same model as the PMD example: two uniform links connected by revolute
joints, released from the initial configuration below under gravity.

+--------+-----------+------------------+------------------+
| Body   | Mass [kg] | Izz [kg m^2]     | Initial angle    |
+--------+-----------+------------------+------------------+
| link_1 | 1.756261  | 0.011854         | -12.9946 deg     |
| link_2 | 2.692381  | 0.042171         | +13.7698 deg     |
+--------+-----------+------------------+------------------+

Units: m, kg, N.
"""

from __future__ import annotations

import numpy as np

from modyn.constraints import RevoluteJoint
from modyn.forces import Gravity
from modyn.model import Ground, Marker, RigidBody
from modyn.solver import SpatialMultibodyModel

T_FINAL = 35.0
DT = 1e-3

_L1, _L2 = 0.250, 0.400
_THETA_DEG = (-12.9946167919, 13.7697807428)
_M1, _J1 = 1.756261, 0.011854
_M2, _J2 = 2.692381, 0.042171


def _body(name: str, mass: float, izz: float) -> RigidBody:
    # Planar motion about z: Ixx and Iyy do not enter the equations.
    return RigidBody(mass=mass, inertia=np.diag([izz, izz, izz]), name=name)


def build_model() -> tuple[SpatialMultibodyModel, dict]:
    """Return the model and a dictionary with its bodies and joints."""
    th1, th2 = np.deg2rad(_THETA_DEG)
    ground = Ground()
    link_1 = _body("link_1", _M1, _J1)
    link_2 = _body("link_2", _M2, _J2)

    g_l1 = Marker(ground, position=(0.0, 0.0, 0.0))
    l1_g = Marker(link_1, position=(-_L1 / 2, 0.0, 0.0))
    l1_l2 = Marker(link_1, position=(_L1 / 2, 0.0, 0.0))
    l2_l1 = Marker(link_2, position=(-_L2 / 2, 0.0, 0.0))

    j_g_l1 = RevoluteJoint(g_l1, l1_g, q0=[th1], name="ground_link_1_joint")
    j_l1_l2 = RevoluteJoint(l1_l2, l2_l1, q0=[th2 - th1], name="link_1_link_2_joint")

    model = SpatialMultibodyModel()
    model.add_body(link_1)
    model.add_body(link_2)
    model.add_joint(j_g_l1)
    model.add_joint(j_l1_l2)
    model.add_force(Gravity((0.0, -9.80665, 0.0)))
    return model, {"bodies": [link_1, link_2], "joints": [j_g_l1, j_l1_l2]}


if __name__ == "__main__":
    model, parts = build_model()
    result = model.solve((0.0, T_FINAL), DT, progress=True)
    print(f"success: {result.success}")
    print(f"max constraint violation: {result.info['max_constraint_violation']:.2e}")
