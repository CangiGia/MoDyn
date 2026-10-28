"""Tests for the redundancy analysis."""

import numpy as np

from modyn.dynamics.redundancy import select_independent_rows


def test_full_rank_keeps_every_row():
    rng = np.random.default_rng(0)
    D = rng.normal(size=(4, 9))
    sel = select_independent_rows(D)
    assert sel.rank == 4
    np.testing.assert_array_equal(sel.rows, np.arange(4))
    assert sel.n_redundant == 0


def test_duplicated_and_combined_rows_are_dropped():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(3, 8))
    D = np.vstack([base, base[0], 2.0 * base[1] - 0.5 * base[2]])
    sel = select_independent_rows(D)
    assert sel.rank == 3
    assert sel.n_redundant == 2
    # the kept rows still span the whole row space
    assert np.linalg.matrix_rank(D[sel.rows]) == 3
    np.testing.assert_array_equal(np.sort(np.concatenate((sel.rows, sel.dropped))), np.arange(5))


def test_insensitive_to_row_scaling():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(3, 8))
    D = np.vstack([base, 1e-6 * base[0]])
    sel = select_independent_rows(D)
    assert sel.rank == 3 and sel.n_redundant == 1


def test_zero_rows_and_empty_input():
    D = np.zeros((2, 5))
    D[0, 1] = 1.0
    sel = select_independent_rows(D)
    assert sel.rank == 1 and list(sel.rows) == [0] and list(sel.dropped) == [1]
    empty = select_independent_rows(np.zeros((0, 4)))
    assert empty.rank == 0 and empty.rows.size == 0
    assert select_independent_rows(np.zeros((3, 4))).rank == 0
