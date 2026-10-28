"""Unit tests for ``modyn.constraints.universal``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.constraints import UniversalJoint
from modyn.constraints.base import Constraint
from modyn.math import quaternions as _q  # noqa: F401  (kept for parity with siblings)
from modyn.math.rotations import roty
from modyn.model.ground import Ground
from modyn.model.marker import Marker

from ._joint_helpers import (
    assert_gamma_matches_phi_ddot,
    assert_phi_q_matches_fd,
    make_body,
    random_quat,
)


class TestUniversalJoint:
    def test_is_a_constraint(self):
        g = Ground()
        b = make_body()
        # Markers oriented so z-axes are initially perpendicular:
        # mi.z = e_z; mj.z = e_x (via a -90° rotation about y, which sends
        # e_z → e_x).
        Ry = roty(-np.pi / 2.0)
        j = UniversalJoint(Marker(g), Marker(b, orientation=Ry))
        assert isinstance(j, Constraint)
        assert j.n_eq == 4

    def test_shapes(self):
        Ry = roty(-np.pi / 2.0)
        b1 = make_body()
        b2 = make_body()
        j = UniversalJoint(Marker(b1), Marker(b2, orientation=Ry))
        assert j.phi().shape == (4,)
        blocks = j.phi_q()
        assert len(blocks) == 2
        for _b, J_R, J_p in blocks:
            assert J_R.shape == (4, 3)
            assert J_p.shape == (4, 4)
        assert j.gamma().shape == (4,)

    def test_residual_zero_when_pins_perpendicular_and_coincident(self):
        Ry = roty(-np.pi / 2.0)
        b1 = make_body(R=(1.0, 2.0, 3.0))
        b2 = make_body(R=(1.0, 2.0, 3.0))
        j = UniversalJoint(Marker(b1), Marker(b2, orientation=Ry))
        np.testing.assert_allclose(j.phi(), 0.0, atol=1e-15)

    def test_residual_nonzero_when_pins_not_perpendicular(self):
        # Identity-orientation markers ⇒ both pin axes = e_z ⇒ z_i · z_j = 1.
        b1 = make_body()
        b2 = make_body()
        j = UniversalJoint(Marker(b1), Marker(b2))
        phi = j.phi()
        np.testing.assert_allclose(phi[:3], 0.0, atol=1e-15)  # coincident origins
        np.testing.assert_allclose(phi[3], 1.0, atol=1e-15)

    def test_residual_picks_up_marker_offset(self):
        Ry = roty(-np.pi / 2.0)
        b1 = make_body(R=(0.0, 0.0, 0.0))
        b2 = make_body(R=(0.3, -0.1, 0.2))
        j = UniversalJoint(Marker(b1), Marker(b2, orientation=Ry))
        phi = j.phi()
        np.testing.assert_allclose(phi[:3], (-0.3, 0.1, -0.2), atol=1e-15)

    def test_ground_block_is_omitted(self):
        Ry = roty(-np.pi / 2.0)
        g = Ground()
        b = make_body()
        j = UniversalJoint(Marker(g), Marker(b, orientation=Ry))
        blocks = j.phi_q()
        assert len(blocks) == 1
        assert blocks[0][0] is b

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_phi_q_finite_difference(self, seed):
        rng = np.random.default_rng(seed)
        Ry = roty(-np.pi / 2.0)
        b1 = make_body(R=rng.standard_normal(3), p=random_quat(rng))
        b2 = make_body(R=rng.standard_normal(3), p=random_quat(rng))
        m1 = Marker(b1, position=rng.standard_normal(3))
        m2 = Marker(b2, position=rng.standard_normal(3), orientation=Ry)
        j = UniversalJoint(m1, m2)
        assert_phi_q_matches_fd(j, (b1, b2))

    @pytest.mark.parametrize("seed", [0, 3, 11])
    def test_gamma_matches_phi_ddot(self, seed):
        rng = np.random.default_rng(seed)
        Ry = roty(-np.pi / 2.0)
        b1 = make_body(
            R=rng.standard_normal(3),
            p=random_quat(rng),
            v=rng.standard_normal(3),
            w=rng.standard_normal(3),
        )
        b2 = make_body(
            R=rng.standard_normal(3),
            p=random_quat(rng),
            v=rng.standard_normal(3),
            w=rng.standard_normal(3),
        )
        m1 = Marker(b1, position=rng.standard_normal(3))
        m2 = Marker(b2, position=rng.standard_normal(3), orientation=Ry)
        j = UniversalJoint(m1, m2)
        assert_gamma_matches_phi_ddot(j, (b1, b2))

    def test_marker_accessors(self):
        Ry = roty(-np.pi / 2.0)
        b1 = make_body()
        b2 = make_body()
        m1 = Marker(b1)
        m2 = Marker(b2, orientation=Ry)
        j = UniversalJoint(m1, m2, name="hooke")
        assert j.marker_i is m1
        assert j.marker_j is m2
        assert j.name == "hooke"
