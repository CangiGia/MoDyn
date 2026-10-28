"""Unit tests for ``modyn.constraints.prismatic``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.constraints import PrismaticJoint
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


class TestPrismaticJoint:
    def test_is_a_constraint(self):
        g = Ground()
        b = make_body()
        j = PrismaticJoint(Marker(g), Marker(b))
        assert isinstance(j, Constraint)
        assert j.n_eq == 5

    def test_shapes(self):
        b1 = make_body()
        b2 = make_body(R=(0.0, 0.0, 1.0))
        j = PrismaticJoint(Marker(b1), Marker(b2, position=(0.0, 0.0, -0.5)))
        assert j.phi().shape == (5,)
        blocks = j.phi_q()
        assert len(blocks) == 2
        for _b, J_R, J_p in blocks:
            assert J_R.shape == (5, 3)
            assert J_p.shape == (5, 4)
        assert j.gamma().shape == (5,)

    def test_residual_zero_when_aligned_and_translated_along_axis(self):
        # Both bodies have identity orientation; marker_j sits directly above
        # marker_i along the common z-axis ⇒ all 5 residuals zero.
        b1 = make_body(R=(0.0, 0.0, 0.0))
        b2 = make_body(R=(0.0, 0.0, 2.5))
        j = PrismaticJoint(Marker(b1), Marker(b2))
        np.testing.assert_allclose(j.phi(), 0.0, atol=1e-15)

    def test_residual_nonzero_under_rotation_about_slide_axis(self):
        # Rotating body 2 about its z-axis breaks the "no rotation about axis"
        # equation (row 3 = x_i · y_j = -sin(θ)) while keeping axis-parallel
        # and translation rows zero.
        angle = np.deg2rad(30.0)
        b1 = make_body()
        b2 = make_body(R=(0.0, 0.0, 1.0), p=_q.from_axis_angle((0.0, 0.0, 1.0), angle))
        j = PrismaticJoint(Marker(b1), Marker(b2))
        phi = j.phi()
        np.testing.assert_allclose(phi[:2], 0.0, atol=1e-15)
        np.testing.assert_allclose(phi[2], -np.sin(angle), atol=1e-15)
        np.testing.assert_allclose(phi[3:], 0.0, atol=1e-15)

    def test_residual_nonzero_under_off_axis_translation(self):
        # Translate body 2 by (0.7, 0, 0): the in-plane offset is picked up
        # by the dot-2 row x_i · d.
        b1 = make_body()
        b2 = make_body(R=(0.7, 0.0, 0.5))
        j = PrismaticJoint(Marker(b1), Marker(b2))
        phi = j.phi()
        np.testing.assert_allclose(phi[:3], 0.0, atol=1e-15)
        np.testing.assert_allclose(phi[3], 0.7, atol=1e-15)
        np.testing.assert_allclose(phi[4], 0.0, atol=1e-15)

    def test_ground_block_is_omitted(self):
        g = Ground()
        b = make_body()
        j = PrismaticJoint(Marker(g), Marker(b))
        blocks = j.phi_q()
        assert len(blocks) == 1
        body, _J_R, _J_p = blocks[0]
        assert body is b

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_phi_q_finite_difference(self, seed):
        rng = np.random.default_rng(seed)
        b1 = make_body(R=rng.standard_normal(3), p=random_quat(rng))
        b2 = make_body(R=rng.standard_normal(3), p=random_quat(rng))
        m1 = Marker(b1, position=rng.standard_normal(3))
        m2 = Marker(b2, position=rng.standard_normal(3))
        j = PrismaticJoint(m1, m2)
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
        j = PrismaticJoint(m1, m2)
        assert_gamma_matches_phi_ddot(j, (b1, b2))

    def test_marker_accessors(self):
        b1 = make_body()
        b2 = make_body()
        m1 = Marker(b1)
        m2 = Marker(b2)
        j = PrismaticJoint(m1, m2, name="slider")
        assert j.marker_i is m1
        assert j.marker_j is m2
        assert j.name == "slider"
