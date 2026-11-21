"""The Burgers reference solver must be trustworthy before anything is compared with it."""

import numpy as np
import pytest

from physics_surrogates.models.fno import heat_solution, random_periodic_fields
from physics_surrogates.solvers import burgers_solve

NU = 0.01


def _heat_gap(eps: float) -> float:
    u0 = eps * random_periodic_fields(4, 128, seed=1)
    got = burgers_solve(u0, NU, 0.5, dt=1e-3)
    want = heat_solution(u0, NU, 0.5)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def test_linear_limit_matches_exact_heat_solution() -> None:
    """The gap to the exact heat solution is the nonlinear term: it must scale as O(eps)."""
    g4, g6 = _heat_gap(1e-4), _heat_gap(1e-6)
    assert g6 < 1e-5
    assert 50 < g4 / g6 < 200  # amplitude ratio is 100 -> linear scaling


def test_mean_is_conserved() -> None:
    u0 = random_periodic_fields(3, 128, seed=2) + 0.3
    out = burgers_solve(u0, NU, 0.5)
    assert np.allclose(out.mean(-1), u0.mean(-1), atol=1e-10)


def test_energy_does_not_grow() -> None:
    u0 = random_periodic_fields(3, 128, seed=3)
    out = burgers_solve(u0, NU, 0.5)
    assert np.all((out**2).mean(-1) <= (u0**2).mean(-1) + 1e-12)


def test_time_step_convergence() -> None:
    u0 = random_periodic_fields(2, 256, seed=4)
    ref = burgers_solve(u0, NU, 0.5, dt=1e-4)
    e_coarse = np.max(np.abs(burgers_solve(u0, NU, 0.5, dt=4e-3) - ref))
    e_fine = np.max(np.abs(burgers_solve(u0, NU, 0.5, dt=2e-3) - ref))
    assert e_fine < e_coarse / 8  # at least ~3rd order observed (RK4 nominal: 16x)
    assert e_fine < 1e-6


def test_spatial_convergence_resolved_front() -> None:
    u0_hi = random_periodic_fields(2, 512, seed=5)
    hi = burgers_solve(u0_hi, NU, 0.5, dt=5e-4)
    lo = burgers_solve(u0_hi[:, ::2], NU, 0.5, dt=5e-4)  # 256 points, same smooth ICs
    assert np.max(np.abs(lo - hi[:, ::2])) < 1e-6


def test_hits_final_time_exactly_for_awkward_dt() -> None:
    u0 = random_periodic_fields(1, 64, seed=6)
    a = burgers_solve(u0, NU, 0.5, dt=0.0033)
    b = burgers_solve(u0, NU, 0.5, dt=1e-3)
    assert np.max(np.abs(a - b)) < 1e-6


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        burgers_solve(np.zeros((1, 8)), -1.0, 1.0)
