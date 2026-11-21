"""Cross-checks for the analytical reference solution.

``scripts/run_benchmark_real.py`` scores the PINN against ``cole_hopf_burgers``.
If that reference were wrong, every reported number would be wrong with it, so it
is checked here in three independent ways: the initial condition, the boundary
conditions and odd symmetry are exact properties of the problem, and the profile
is compared with a Fourier pseudo-spectral solution of the same PDE.
"""

import numpy as np
import pytest

from physics_surrogates.reference import (
    burgers_initial_condition,
    burgers_spectral_reference,
    cole_hopf_burgers,
)

NU = 0.01 / np.pi


def test_initial_condition_at_t_zero():
    x = np.linspace(-1, 1, 41)
    np.testing.assert_allclose(cole_hopf_burgers(x, 0.0, NU), -np.sin(np.pi * x), atol=1e-14)


def test_initial_condition_helper_matches_formula():
    x = np.linspace(-1, 1, 7)
    np.testing.assert_allclose(burgers_initial_condition(x), -np.sin(np.pi * x))


def test_scalar_and_array_inputs_agree():
    scalar = cole_hopf_burgers(0.37, 0.5, NU)
    array = cole_hopf_burgers(np.array([0.37]), 0.5, NU)
    assert isinstance(scalar, float)
    assert array.shape == (1,)
    assert scalar == pytest.approx(float(array[0]))


def test_dirichlet_boundaries_vanish():
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        values = cole_hopf_burgers(np.array([-1.0, 1.0]), t, NU)
        assert np.allclose(values, 0.0, atol=1e-12), (t, values)


def test_solution_is_odd_in_x():
    x = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(
        cole_hopf_burgers(-x, 0.5, NU),
        -cole_hopf_burgers(x, 0.5, NU),
        atol=1e-10,
    )


def test_shock_steepens_with_time():
    x = np.linspace(-1, 1, 401)
    early = cole_hopf_burgers(x, 0.25, NU)
    late = cole_hopf_burgers(x, 1.0, NU)
    # A steeper front means a larger spatial derivative, not a larger amplitude.
    assert np.max(np.abs(np.gradient(late, x))) > np.max(np.abs(np.gradient(early, x)))
    assert np.max(np.abs(late)) < np.max(np.abs(early))


@pytest.mark.parametrize("t_final", [0.5, 1.0])
def test_matches_independent_spectral_solver(t_final):
    """Cole-Hopf quadrature against a pseudo-spectral solve of the same PDE."""
    x, u_spectral = burgers_spectral_reference(t_final, NU, n_x=384, n_steps=8000)
    interior = np.abs(x) < 0.995  # stay away from the Dirichlet end points
    u_cole_hopf = cole_hopf_burgers(x[interior], t_final, NU)

    rel = np.linalg.norm(u_cole_hopf - u_spectral[interior]) / np.linalg.norm(u_cole_hopf)
    assert rel < 5e-3, f"t={t_final}: relative disagreement {rel:.3e}"


def test_spectral_solver_converges_under_refinement():
    """Two grids of the independent solver must approach each other."""
    x_coarse, u_coarse = burgers_spectral_reference(0.5, NU, n_x=256, n_steps=4000)
    x_fine, u_fine = burgers_spectral_reference(0.5, NU, n_x=512, n_steps=8000)

    # The grids differ, so compare on the fine grid away from the periodic seam.
    u_coarse_on_fine = np.interp(x_fine, x_coarse, u_coarse)
    interior = np.abs(x_fine) < 0.9
    rel = np.linalg.norm(u_coarse_on_fine[interior] - u_fine[interior]) / np.linalg.norm(
        u_fine[interior]
    )
    assert rel < 5e-2, rel
