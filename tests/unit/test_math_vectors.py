"""Tests for ``modyn.math.vectors``."""

from __future__ import annotations

import numpy as np
import pytest

from modyn.math.vectors import as_vec3, cross, skew, unskew

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "src",
    [
        [1.0, 2.0, 3.0],
        (1, 2, 3),
        np.array([1, 2, 3]),
        np.array([[1], [2], [3]]),
        np.array([[1, 2, 3]]),
    ],
)
def test_as_vec3_shapes(src):
    v = as_vec3(src)
    assert v.shape == (3,)
    assert v.dtype == np.float64
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])


def test_as_vec3_returns_copy():
    src = np.array([1.0, 2.0, 3.0])
    v = as_vec3(src)
    v[0] = 99.0
    assert src[0] == 1.0


def test_as_vec3_invalid_length():
    with pytest.raises(ValueError):
        as_vec3([1.0, 2.0])


def test_skew_is_antisymmetric():
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.standard_normal(3)
        S = skew(v)
        np.testing.assert_allclose(S, -S.T, atol=1e-15)
        assert np.trace(S) == pytest.approx(0.0, abs=1e-15)


def test_skew_acts_as_cross_product():
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = rng.standard_normal(3)
        w = rng.standard_normal(3)
        np.testing.assert_allclose(skew(v) @ w, np.cross(v, w), atol=1e-14)


def test_unskew_is_inverse_of_skew():
    rng = np.random.default_rng(2)
    for _ in range(20):
        v = rng.standard_normal(3)
        np.testing.assert_allclose(unskew(skew(v)), v, atol=1e-15)


def test_unskew_uses_antisymmetric_part():
    # If M is contaminated by a symmetric perturbation, unskew should
    # recover the antisymmetric component robustly.
    v = np.array([0.5, -1.5, 2.0])
    sym = np.array([[1.0, 0.2, -0.3], [0.2, 0.5, 0.7], [-0.3, 0.7, -2.0]])
    M = skew(v) + sym
    np.testing.assert_allclose(unskew(M), v, atol=1e-15)


def test_unskew_invalid_shape():
    with pytest.raises(ValueError):
        unskew(np.zeros((2, 2)))


def test_cross_matches_numpy():
    rng = np.random.default_rng(3)
    for _ in range(10):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        np.testing.assert_allclose(cross(a, b), np.cross(a, b), atol=1e-15)
