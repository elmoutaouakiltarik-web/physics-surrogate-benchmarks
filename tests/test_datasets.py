"""
Tests for benchmark datasets.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_surrogates.datasets.ghia_cavity import (
    cavity_bc_mask,
    cavity_mms_manufactured_solution,
    evaluate_cavity_solution,
    generate_cavity_mesh,
    get_ghia_reference,
)


class TestGhiaReference:
    """Test Ghia et al. reference data."""

    def test_ghia_re100(self):
        """Test Ghia reference for Re=100."""
        ref = get_ghia_reference(100)
        assert ref.re == 100
        assert len(ref.y_probe) == 10
        assert len(ref.u_center) == 10
        assert ref.u_max > 0
        assert ref.v_max > 0

    def test_ghia_re400(self):
        """Test Ghia reference for Re=400."""
        ref = get_ghia_reference(400)
        assert ref.re == 400
        assert len(ref.y_probe) > 0

    def test_ghia_re1000(self):
        """Test Ghia reference for Re=1000."""
        ref = get_ghia_reference(1000)
        assert ref.re == 1000

    def test_ghia_invalid_re(self):
        """Test error on invalid Reynolds number."""
        with pytest.raises(ValueError):
            get_ghia_reference(999)

    def test_ghia_reference_bounds(self):
        """Test that reference velocities are physically reasonable."""
        ref = get_ghia_reference(100)
        # u should be in [-1, 1] (normalized by lid velocity)
        assert np.all(np.abs(ref.u_center) <= 1.5)
        # v should be smaller than u
        assert np.max(np.abs(ref.v_center)) < np.max(np.abs(ref.u_center))


class TestCavityMesh:
    """Test cavity mesh generation."""

    def test_mesh_generation(self):
        """Test mesh grid generation."""
        x, y, X, Y = generate_cavity_mesh(nx=65, ny=65)
        assert len(x) == 65
        assert len(y) == 65
        assert X.shape == (65, 65)
        assert Y.shape == (65, 65)

    def test_mesh_bounds(self):
        """Test mesh is on unit square."""
        x, y, X, Y = generate_cavity_mesh(nx=50, ny=50)
        assert np.min(x) == 0.0
        assert np.max(x) == 1.0
        assert np.min(y) == 0.0
        assert np.max(y) == 1.0

    def test_mesh_orthogonal(self):
        """Test mesh coordinates are orthogonal (structured grid)."""
        x, y, X, Y = generate_cavity_mesh(nx=20, ny=30)
        # Check that X varies only along columns (constant along rows)
        assert np.allclose(X[:, 0], X[:, 1])  # Same X values in all columns
        # Check that Y varies only along rows (constant along columns)
        assert np.allclose(Y[0, :], Y[1, :])  # Same Y values in all rows


class TestCavityBC:
    """Test boundary condition masks."""

    def test_bc_masks(self):
        """Test BC mask generation."""
        top, bottom, left, right = cavity_bc_mask(nx=65, ny=65)
        assert top.sum() == 65  # Full top row
        assert bottom.sum() == 65  # Full bottom row
        assert left.sum() == 65  # Full left column
        assert right.sum() == 65  # Full right column

    def test_bc_no_overlap(self):
        """Test BC masks don't overlap incorrectly."""
        top, bottom, left, right = cavity_bc_mask(nx=65, ny=65)
        # Corners might overlap, but interior should be distinct
        interior_top = top[1:-1, 1:-1]
        interior_bottom = bottom[1:-1, 1:-1]
        assert not np.any(interior_top & interior_bottom)


class TestMMS:
    """Test Method of Manufactured Solutions."""

    def test_mms_fields(self):
        """Test MMS field generation."""
        x, y, X, Y = generate_cavity_mesh(nx=33, ny=33)
        u, v, p = cavity_mms_manufactured_solution(X, Y)
        assert u.shape == (33, 33)
        assert v.shape == (33, 33)
        assert p.shape == (33, 33)

    def test_mms_boundary_conditions(self):
        """Test MMS satisfies cavity BCs (u=v=0 on walls)."""
        x, y, X, Y = generate_cavity_mesh(nx=20, ny=20)
        u, v, p = cavity_mms_manufactured_solution(X, Y)
        # Bottom wall (y=0): u=0, v=0
        assert np.allclose(u[0, :], 0, atol=1e-10)
        assert np.allclose(v[0, :], 0, atol=1e-10)
        # Top wall (y=1): u=0, v=0
        assert np.allclose(u[-1, :], 0, atol=1e-10)
        assert np.allclose(v[-1, :], 0, atol=1e-10)
        # Left wall (x=0): u=0, v=0
        assert np.allclose(u[:, 0], 0, atol=1e-10)
        assert np.allclose(v[:, 0], 0, atol=1e-10)
        # Right wall (x=1): u=0, v=0
        assert np.allclose(u[:, -1], 0, atol=1e-10)
        assert np.allclose(v[:, -1], 0, atol=1e-10)

    def test_mms_continuity(self):
        """Test MMS fields are smooth (no sharp discontinuities)."""
        x, y, X, Y = generate_cavity_mesh(nx=65, ny=65)
        u, v, p = cavity_mms_manufactured_solution(X, Y)
        # Check gradients are finite
        u_grad = np.gradient(u)
        v_grad = np.gradient(v)
        assert np.all(np.isfinite(u_grad))
        assert np.all(np.isfinite(v_grad))


class TestEvaluation:
    """Test solution evaluation metrics."""

    def test_evaluation_perfect_match(self):
        """Test evaluation with perfect match."""
        u_ref = np.random.randn(100, 100)
        v_ref = np.random.randn(100, 100)
        metrics = evaluate_cavity_solution(u_ref, v_ref, u_ref, v_ref)
        assert metrics["l2_u_relative"] < 1e-10
        assert metrics["l2_v_relative"] < 1e-10
        assert metrics["r2_u"] > 0.9999
        assert metrics["r2_v"] > 0.9999

    def test_evaluation_with_noise(self):
        """Test evaluation with noisy prediction."""
        np.random.seed(42)
        u_ref = np.random.randn(50, 50)
        v_ref = np.random.randn(50, 50)
        u_pred = u_ref + 0.1 * np.random.randn(50, 50)
        v_pred = v_ref + 0.1 * np.random.randn(50, 50)

        metrics = evaluate_cavity_solution(u_pred, v_pred, u_ref, v_ref)
        assert 0 < metrics["l2_u_relative"] < 0.2
        assert 0 < metrics["l2_v_relative"] < 0.2
        assert 0.8 < metrics["r2_u"] < 1.0
        assert 0.8 < metrics["r2_v"] < 1.0

    def test_evaluation_bad_prediction(self):
        """Test evaluation with poor prediction."""
        u_ref = np.ones((50, 50))
        v_ref = np.ones((50, 50))
        u_pred = np.zeros((50, 50))
        v_pred = np.zeros((50, 50))

        metrics = evaluate_cavity_solution(u_pred, v_pred, u_ref, v_ref)
        assert metrics["l2_u_relative"] > 0.5
        assert metrics["r2_u"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
