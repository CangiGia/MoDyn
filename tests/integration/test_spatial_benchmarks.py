"""Integration tests: spatial benchmarks with an independent reference solution."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("casadi")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation" / "analytical"))
import spatial_benchmarks as sb  # noqa: E402

pytestmark = pytest.mark.integration


def test_free_asymmetric_body_matches_euler_equations():
    _, _, metrics = sb.free_rigid_body(t_final=10.0, dt=0.01)
    assert metrics["max |omega' - omega'_ref| [rad/s]"] < 1e-8
    assert metrics["max |dL| / |L|"] < 1e-8
    assert metrics["max |dE| / E"] < 1e-9
    assert metrics["max ||p|| - 1"] < 1e-9


def test_lagrange_top_matches_reference_and_converges_with_order_five():
    coarse = sb.lagrange_top(t_final=2.0, dt=0.01)[2]["max |r_cm - r_cm_ref| [m]"]
    fine = sb.lagrange_top(t_final=2.0, dt=0.005)[2]["max |r_cm - r_cm_ref| [m]"]
    assert fine < 5e-5
    assert coarse / fine > 16.0  # Radau IIA: order 5 (factor ~32 per halving)
