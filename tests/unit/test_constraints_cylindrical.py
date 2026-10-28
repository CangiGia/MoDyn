"""Unit tests for ``modyn.constraints.cylindrical``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.constraints import CylindricalJoint
from modyn.constraints.base import Constraint
from modyn.math import quaternions as _q
from modyn.model.ground import Ground
from modyn.model.marker import Marker

from ._joint_helpers import (
    assert_gamma_matches_phi_ddot,
    assert_phi_q_matches_fd,
    make_body,
    random_quat,
)


class TestCylindricalJoint:
    def test_is_a_constraint(self):
        g = Ground()
        b = make_body()
        j = CylindricalJoint(Marker(g), Marker(b))
        assert isinstance(j, Constraint)
        assert j.n_eq == 4

    def test_shapes(self):
        b1 = make_body()
        b2 = make_body(R=(0.0, 0.0, 1.0))
        j = CylindricalJoint(Marker(b1), Marker(b2))
        assert j.phi().shape == (4,)
        blocks = j.phi_q()
        assert len(blocks) == 2
        for _b, J_R, J_p in blocks:
            assert J_R.shape == (4, 3)
            assert J_p.shape == (4, 4)
        assert j.gamma().shape == (4,)

    def test_residual_zero_under_rotation_about_axis(self):
        # Cylindrical leaves rotation about the common axis free.
        b1 = make_body()
        b2 = make_body(
            R=(0.0, 0.0, 1.7),
            p=_q.from_axis_angle((0.0, 0.0, 1.0), np.deg2rad(45.0)),
        )
        j = CylindricalJoint(Marker(b1), Marker(b2))
        np.testing.assert_allclose(j.phi(), 0.0, atol=1e-14)

    def test_residual_nonzero_under_off_axis_translation(self):
        b1 = make_body()
        b2 = make_body(R=(0.4, 0.2, 0.5))
        j = CylindricalJoint(Marker(b1), Marker(b2))
        phi = j.phi()
        np.testing.assert_allclose(phi[:2], 0.0, atol=1e-15)
        np.testing.assert_allclose(phi[2], 0.4, atol=1e-15)
        np.testing.assert_allclose(phi[3], 0.2, atol=1e-15)

    def test_residual_nonzero_under_axis_tilt(self):
        # Tilting body 2 about x breaks parallelism of z-axes.
        angle = np.deg2rad(20.0)
        b1 = make_body()
        b2 = make_body(p=_q.from_axis_angle((1.0, 0.0, 0.0), angle))
        j = CylindricalJoint(Marker(b1), Marker(b2))
        phi = j.phi()
        np.testing.assert_allclose(phi[0], 0.0, atol=1e-15)  # x_i · z_j
        np.testing.assert_allclose(phi[1], -np.sin(angle), atol=1e-15)

    def test_ground_block_is_omitted(self):
        g = Ground()
        b = make_body()
        j = CylindricalJoint(Marker(g), Marker(b))
        blocks = j.phi_q()
        assert len(blocks) == 1
        assert blocks[0][0] is b

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_phi_q_finite_difference(self, seed):
        rng = np.random.default_rng(seed)
        b1 = make_body(R=rng.standard_normal(3), p=random_quat(rng))
        b2 = make_body(R=rng.standard_normal(3), p=random_quat(rng))
        m1 = Marker(b1, position=rng.standard_normal(3))
        m2 = Marker(b2, position=rng.standard_normal(3))
        j = CylindricalJoint(m1, m2)
        assert_phi_q_matches_fd(j, (b1, b2))

    @pytest.mark.parametrize("seed", [0, 3, 11])
    def test_gamma_matches_phi_ddot(self, seed):
        rng = np.random.default_rng(seed)
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
        m2 = Marker(b2, position=rng.standard_normal(3))
        j = CylindricalJoint(m1, m2)
        assert_gamma_matches_phi_ddot(j, (b1, b2))

    def test_marker_accessors(self):
        b1 = make_body()
        b2 = make_body()
        m1 = Marker(b1)
        m2 = Marker(b2)
        j = CylindricalJoint(m1, m2, name="cyl")
        assert j.marker_i is m1
        assert j.marker_j is m2
        assert j.name == "cyl"
