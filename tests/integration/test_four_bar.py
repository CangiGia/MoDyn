"""Integration test: redundant four-bar linkage (planar mechanism in 3D)."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import four_bar_linkage as fbl  # noqa: E402

pytestmark = pytest.mark.integration


def test_four_bar_assembly_and_redundancy():
    model, parts = fbl.build_model()
    result = model.solve((0.0, 0.2), 1e-3)
    assert result.success
    # 4 revolute joints (20 eqs) + driver on 3 bodies: 3 redundant rows
    assert result.info["dropped_rows"].size == 3
    assert result.info["rank"] == 6 * 3
    assert result.info["max_constraint_violation"] < 1e-9
    assert result.quaternion_norm_error() < 1e-9
    topology = model.topology
    assert len(topology.tree) == 3 and len(topology.cut_joints) == 1 and topology.n_loops == 1


def test_four_bar_motion_is_planar_and_driven():
    model, parts = fbl.build_model()
    result = model.solve((0.0, 1.0), 1e-3)
    for i in range(3):
        np.testing.assert_allclose(result.position(i)[:, 2], 0.0, atol=1e-10)
    # crank angle follows the prescribed law theta(t) = theta0 + omega t
    p = result.quaternion(0)
    theta = 2.0 * np.arctan2(p[:, 3], p[:, 0])
    expected = np.deg2rad(fbl._THETA_DEG[0] + fbl._OMEGA_DEG * result.times)
    np.testing.assert_allclose(np.unwrap(theta), expected, atol=1e-9)
    # out-of-plane reactions are zero (minimum-norm solution)
    for joint in parts["joints"]:
        R = model.reaction(result, joint, side="i")
        np.testing.assert_allclose(R[:, 2], 0.0, atol=1e-9)


def test_four_bar_solvers_agree():
    model_a, parts_a = fbl.build_model()
    model_b, parts_b = fbl.build_model()
    r1 = model_a.solve((0.0, 0.1), 1e-3)
    r2 = model_b.solve((0.0, 0.1), 1e-3, method="generalized_alpha", rho_inf=0.9)
    for i in range(3):
        np.testing.assert_allclose(r1.position(i), r2.position(i), atol=1e-7)
    ra = model_a.reaction(r1, parts_a["joints"][0])
    rb = model_b.reaction(r2, parts_b["joints"][0])
    np.testing.assert_allclose(ra, rb, atol=1e-6)
