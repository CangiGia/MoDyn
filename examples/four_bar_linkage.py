"""Four-bar linkage — planar mechanism simulated in 3D with MoDyn.

Same model as the PMD example: a crank driven at constant angular velocity,
a coupler and a follower, four parallel revolute joints. In 3D the four
joints are redundant (the Jacobian loses rank), which MoDyn detects and
removes automatically.

Bodies are defined only through joint coordinates ``q0`` (the angle of each
tree joint); poses are obtained by the topological assembly and the loop is
closed by Newton.

Units: m, kg, N. Joint order and marker order follow the ADAMS model
(``validation/four-bar linkage/ADAMS``).
"""

from __future__ import annotations

import numpy as np

from modyn.constraints import RevoluteDriver, RevoluteJoint
from modyn.constraints.functions import Linear
from modyn.forces import Gravity
from modyn.model import Ground, Marker, RigidBody
from modyn.solver import SpatialMultibodyModel

T_FINAL = 25.0
DT = 1e-3

_L1, _L2, _L3 = 40.0e-3, 120.0e-3, 80.0e-3  # crank, coupler, follower lengths
_O4 = (0.100, 0.000, 0.000)  # follower pivot
_THETA_DEG = (40.000, 20.298, -122.605)  # absolute angles of the three links
_OMEGA_DEG = -25.0  # crank angular velocity [deg/s]


def _body(name: str, mass: float, izz: float) -> RigidBody:
    # Planar motion about z: Ixx and Iyy do not enter the equations.
    return RigidBody(mass=mass, inertia=np.diag([izz, izz, izz]), name=name)


def build_model() -> tuple[SpatialMultibodyModel, dict]:
    """Return the model and a dictionary with its bodies and joints."""
    th = np.deg2rad(_THETA_DEG)
    ground = Ground()

    link_1 = _body("link_1", 1.1528090607e-02, 7.4696175803e-05)
    link_2 = _body("link_2", 3.1498650607e-02, 0.2078332421e-05)
    link_3 = _body("link_3", 2.1513370607e-02, 0.141264709e-03)

    g_l1 = Marker(ground, position=(0.0, 0.0, 0.0))
    g_l3 = Marker(ground, position=_O4)
    l1_g = Marker(link_1, position=(-_L1 / 2, 0.0, 0.0))
    l1_l2 = Marker(link_1, position=(_L1 / 2, 0.0, 0.0))
    l2_l1 = Marker(link_2, position=(-_L2 / 2, 0.0, 0.0))
    l2_l3 = Marker(link_2, position=(_L2 / 2, 0.0, 0.0))
    l3_l2 = Marker(link_3, position=(-_L3 / 2, 0.0, 0.0))
    l3_g = Marker(link_3, position=(_L3 / 2, 0.0, 0.0))

    j_g_l1 = RevoluteJoint(g_l1, l1_g, q0=[th[0]], name="ground_link_1_joint")
    j_l1_l2 = RevoluteJoint(l1_l2, l2_l1, q0=[th[1] - th[0]], name="link_1_link_2_joint")
    j_l2_l3 = RevoluteJoint(l2_l3, l3_l2, q0=[th[2] - th[1]], name="link_2_link_3_joint")
    j_l3_g = RevoluteJoint(l3_g, g_l3, q0=[th[2]], name="link_3_ground_joint")
    motion = RevoluteDriver(
        g_l1, l1_g, Linear(slope=np.deg2rad(_OMEGA_DEG), intercept=th[0]), name="crank_motion"
    )

    model = SpatialMultibodyModel()
    for body in (link_1, link_2, link_3):
        model.add_body(body)
    for joint in (j_g_l1, j_l1_l2, j_l2_l3, j_l3_g, motion):
        model.add_joint(joint)
    model.add_force(Gravity((0.0, -9.80665, 0.0)))

    parts = {
        "bodies": [link_1, link_2, link_3],
        "joints": [j_g_l1, j_l1_l2, j_l2_l3, j_l3_g],
        "motion": motion,
    }
    return model, parts


if __name__ == "__main__":
    model, parts = build_model()
    result = model.solve((0.0, T_FINAL), DT, method="radau", progress=True)
    print(f"success: {result.success}")
    print(f"redundant rows removed: {result.info['dropped_rows'].size}")
    print(f"max constraint violation: {result.info['max_constraint_violation']:.2e}")
    print(f"max |p| - 1: {result.info['quaternion_norm_error']:.2e}")
    print(f"wall time: {result.info['wall_time']:.1f} s")
