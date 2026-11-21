"""Tests for the cavity benchmark data and helpers."""

import numpy as np
import pytest

from physics_surrogates.datasets.ghia_cavity import (
    AVAILABLE_REYNOLDS,
    available_reynolds,
    cavity_bc_mask,
    cavity_mms_manufactured_solution,
    cavity_mms_streamfunction,
    evaluate_cavity_solution,
    generate_cavity_mesh,
    get_ghia_reference,
    ghia_centerline_errors,
)


class TestGhiaReference:
    """The tables must be the published ones, not a smoothed lookalike."""

    @pytest.mark.parametrize("re", AVAILABLE_REYNOLDS)
    def test_grids_and_lengths(self, re):
        ref = get_ghia_reference(re)
        assert ref.re == re
        assert len(ref.y_probe) == len(ref.u_center) == 17
        assert len(ref.x_probe) == len(ref.v_center) == 17
        # Ghia's probes are non-uniform, clustered at the walls.
        assert ref.y_probe[0] == 1.0 and ref.y_probe[-1] == 0.0
        assert np.all(np.diff(ref.y_probe) < 0)

    @pytest.mark.parametrize(
        ("re", "y_at_min", "u_min"),
        [(100, 0.4531, -0.21090), (400, 0.2813, -0.32726), (1000, 0.1719, -0.38289)],
    )
    def test_published_minimum_of_u(self, re, y_at_min, u_min):
        ref = get_ghia_reference(re)
        assert ref.u_center.min() == pytest.approx(u_min, abs=1e-5)
        assert ref.y_probe[int(np.argmin(ref.u_center))] == pytest.approx(y_at_min)

    @pytest.mark.parametrize(
        ("re", "u_near_lid", "v_peak"),
        [(100, 0.84123, 0.17527), (400, 0.75837, 0.30203), (1000, 0.65928, 0.37095)],
    )
    def test_published_extrema(self, re, u_near_lid, v_peak):
        ref = get_ghia_reference(re)
        assert ref.u_center[1] == pytest.approx(u_near_lid, abs=1e-5)
        assert ref.v_center.max() == pytest.approx(v_peak, abs=1e-5)

    def test_lid_and_wall_values(self):
        for re in AVAILABLE_REYNOLDS:
            ref = get_ghia_reference(re)
            assert ref.u_center[0] == pytest.approx(1.0)  # moving lid
            assert ref.u_center[-1] == 0.0  # no-slip bottom
            assert ref.v_center[0] == 0.0 and ref.v_center[-1] == 0.0

    def test_backflow_appears_with_reynolds(self):
        """Deeper cavities at higher Re produce a stronger recirculation."""
        minima = [get_ghia_reference(re).u_center.min() for re in AVAILABLE_REYNOLDS]
        assert minima == sorted(minima, reverse=True)

    @pytest.mark.parametrize("re", [999, 100.5, 50])
    def test_invalid_reynolds(self, re):
        with pytest.raises(ValueError):
            get_ghia_reference(re)

    def test_available_reynolds_helper(self):
        assert available_reynolds() == AVAILABLE_REYNOLDS


class TestCavityMesh:
    def test_mesh_generation(self):
        x, y, X, Y = generate_cavity_mesh(nx=65, ny=65)
        assert len(x) == len(y) == 65
        assert X.shape == Y.shape == (65, 65)

    def test_mesh_bounds(self):
        x, y, _, _ = generate_cavity_mesh(nx=50, ny=50)
        assert x[0] == y[0] == 0.0
        assert x[-1] == y[-1] == 1.0

    def test_mesh_is_a_structured_grid(self):
        _, _, X, Y = generate_cavity_mesh(nx=20, ny=30)
        # X is constant along y, Y is constant along x.
        assert np.allclose(X[:, [0]], X)
        assert np.allclose(Y[[0], :], Y)


class TestCavityBC:
    def test_masks_cover_each_wall_once(self):
        top, bottom, left, right = cavity_bc_mask(nx=65, ny=65)
        assert top.sum() == bottom.sum() == left.sum() == right.sum() == 65
        assert top[-1, :].all() and bottom[0, :].all()
        assert left[:, 0].all() and right[:, -1].all()

    def test_interior_is_unmasked(self):
        top, bottom, left, right = cavity_bc_mask(nx=33, ny=33)
        interior = np.s_[1:-1, 1:-1]
        assert not (top & bottom)[interior].any()
        assert not (left & right)[interior].any()


class TestMMS:
    def test_field_shapes(self):
        _, _, X, Y = generate_cavity_mesh(nx=33, ny=33)
        u, v, p = cavity_mms_manufactured_solution(X, Y)
        assert u.shape == v.shape == p.shape == (33, 33)

    def test_no_slip_on_all_four_walls(self):
        _, _, X, Y = generate_cavity_mesh(nx=21, ny=21)
        u, v, _ = cavity_mms_manufactured_solution(X, Y)
        for edge in (u[0, :], u[-1, :], u[:, 0], u[:, -1], v[0, :], v[-1, :], v[:, 0], v[:, -1]):
            assert np.allclose(edge, 0.0, atol=1e-12)

    def test_is_divergence_free(self):
        """Central differences on the interior must reproduce div(u) = 0."""
        x, y, X, Y = generate_cavity_mesh(nx=257, ny=257)
        u, v, _ = cavity_mms_manufactured_solution(X, Y)
        div = np.gradient(u, x, axis=0) + np.gradient(v, y, axis=1)
        interior = div[1:-1, 1:-1]
        scale = np.abs(np.gradient(u, x, axis=0)).max()
        assert np.abs(interior).max() < 5e-4 * scale

    def test_streamfunction_matches_velocity(self):
        x, y, X, Y = generate_cavity_mesh(nx=257, ny=257)
        u, v, _ = cavity_mms_manufactured_solution(X, Y)
        psi = cavity_mms_streamfunction(X, Y)
        # Central differences on a smooth field: agreement to truncation error.
        np.testing.assert_allclose(np.gradient(psi, y, axis=1), u, atol=5e-3)
        np.testing.assert_allclose(-np.gradient(psi, x, axis=0), v, atol=5e-3)


class TestEvaluation:
    def test_perfect_match(self):
        rng = np.random.default_rng(0)
        u = rng.standard_normal((40, 40))
        v = rng.standard_normal((40, 40))
        metrics = evaluate_cavity_solution(u, v, u, v)
        assert metrics["l2_u_relative"] < 1e-12
        assert metrics["l2_v_relative"] < 1e-12
        assert metrics["r2_u"] > 0.999999

    def test_known_noise_level(self):
        rng = np.random.default_rng(42)
        u_ref = rng.standard_normal((50, 50))
        v_ref = rng.standard_normal((50, 50))
        u_pred = u_ref + 0.1 * rng.standard_normal((50, 50))
        v_pred = v_ref + 0.1 * rng.standard_normal((50, 50))
        metrics = evaluate_cavity_solution(u_pred, v_pred, u_ref, v_ref)
        assert 0 < metrics["l2_u_relative"] < 0.2
        assert 0.8 < metrics["r2_u"] < 1.0
        assert metrics["max_error_u"] > 0

    def test_poor_prediction(self):
        u_ref = np.ones((20, 20))
        zeros = np.zeros((20, 20))
        metrics = evaluate_cavity_solution(zeros, zeros, u_ref, u_ref)
        assert metrics["l2_u_relative"] == pytest.approx(1.0)
        assert metrics["r2_u"] < 0

    def test_missing_reference_returns_empty(self):
        assert evaluate_cavity_solution(np.zeros((5, 5)), np.zeros((5, 5))) == {}

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            evaluate_cavity_solution(np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5, 6)), np.zeros((5, 5)))


class TestGhiaComparison:
    def test_exact_profile_field_has_near_zero_error(self):
        """A field built from the Ghia profiles must score ~0 on both centrelines.

        The residual is grid interpolation only: the field is piecewise linear
        between probes, so a 401-point grid reproduces it to well under a percent.
        """
        x, y, _, _ = generate_cavity_mesh(nx=401, ny=401)
        ref = get_ghia_reference(400)
        # Ghia lists the probes from the lid downwards, so reverse before interp.
        u_line = np.interp(y, ref.y_probe[::-1], ref.u_center[::-1])
        v_line = np.interp(x, ref.x_probe[::-1], ref.v_center[::-1])
        u = np.tile(u_line, (len(x), 1))
        v = np.tile(v_line[:, None], (1, len(y)))

        errors = ghia_centerline_errors(u, v, x, y, re=400)
        assert errors["u_centerline_l2_percent"] < 1.0
        assert errors["v_centerline_l2_percent"] < 1.0

    def test_perturbed_field_reports_error(self):
        x, y, _, _ = generate_cavity_mesh(nx=81, ny=81)
        errors = ghia_centerline_errors(np.zeros((81, 81)), np.zeros((81, 81)), x, y, re=100)
        assert errors["u_centerline_l2_percent"] > 50
        assert errors["v_centerline_l2_percent"] > 50

    def test_shape_validation(self):
        x, y, _, _ = generate_cavity_mesh(nx=41, ny=41)
        with pytest.raises(ValueError):
            ghia_centerline_errors(np.zeros((40, 41)), np.zeros((41, 41)), x, y, re=100)
