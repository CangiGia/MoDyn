"""Tests for ``modyn.math.transforms``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.math import quaternions as q
from modyn.math.transforms import Transform, from_homogeneous

pytestmark = pytest.mark.unit


def _random_transform(rng: np.random.Generator) -> Transform:
    v = rng.standard_normal(4)
    p = v / np.linalg.norm(v)
    t = rng.standard_normal(3)
    return Transform.from_quat_translation(p, t)


def test_identity_apply_is_noop():
    rng = np.random.default_rng(0)
    T = Transform.identity()
    for _ in range(5):
        v = rng.standard_normal(3)
        np.testing.assert_allclose(T.apply(v), v, atol=1e-15)


def test_apply_matches_homogeneous():
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = _random_transform(rng)
        v = rng.standard_normal(3)
        v_h = np.append(v, 1.0)
        expected = (T.as_matrix() @ v_h)[:3]
        np.testing.assert_allclose(T.apply(v), expected, atol=1e-13)


def test_composition_associative_and_consistent_with_matrix():
    rng = np.random.default_rng(2)
    for _ in range(15):
        T1, T2 = _random_transform(rng), _random_transform(rng)
        T12 = T1 @ T2
        np.testing.assert_allclose(T12.as_matrix(), T1.as_matrix() @ T2.as_matrix(), atol=1e-12)
        # Apply via composition equals chain of applications.
        v = rng.standard_normal(3)
        np.testing.assert_allclose(T12.apply(v), T1.apply(T2.apply(v)), atol=1e-13)


def test_inverse_is_two_sided():
    rng = np.random.default_rng(3)
    for _ in range(15):
        T = _random_transform(rng)
        I = Transform.identity()
        np.testing.assert_allclose((T @ T.inverse()).as_matrix(), I.as_matrix(), atol=1e-12)
        np.testing.assert_allclose((T.inverse() @ T).as_matrix(), I.as_matrix(), atol=1e-12)


def test_homogeneous_round_trip():
    rng = np.random.default_rng(4)
    for _ in range(15):
        T = _random_transform(rng)
        H = T.as_matrix()
        T_rec = from_homogeneous(H)
        np.testing.assert_allclose(T_rec.as_matrix(), H, atol=1e-13)


def test_R_property_matches_quaternion():
    rng = np.random.default_rng(5)
    for _ in range(10):
        T = _random_transform(rng)
        np.testing.assert_allclose(T.R, q.A(T.p), atol=1e-15)


def test_to_homogeneous_invalid_inputs():
    with pytest.raises(ValueError):
        from_homogeneous(np.eye(3))
