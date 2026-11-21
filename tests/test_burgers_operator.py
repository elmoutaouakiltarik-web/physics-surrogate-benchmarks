"""Fast sanity tests for the Burgers operator-learning pipeline (not the full benchmark)."""

import numpy as np
import torch

from physics_surrogates.models.burgers_operator import (
    MLPOperator,
    build_fno,
    count_params,
    evaluate_model,
    identity_error,
    make_dataset,
    train_operator,
)


def test_dataset_shapes_and_determinism() -> None:
    a0, at = make_dataset(5, 32, 0.02, 0.5, seed=3, amplitude=2.0)
    b0, bt = make_dataset(5, 32, 0.02, 0.5, seed=3, amplitude=2.0)
    assert a0.shape == at.shape == (5, 32)
    assert np.array_equal(at, bt)


def test_same_function_on_different_grids() -> None:
    _, lo = make_dataset(3, 64, 0.02, 0.5, seed=4, amplitude=2.0)
    _, hi = make_dataset(3, 128, 0.02, 0.5, seed=4, amplitude=2.0)
    assert np.max(np.abs(hi[:, ::2] - lo)) / np.max(np.abs(hi)) < 1e-2


def test_training_reduces_error_and_fno_is_grid_independent() -> None:
    u0, ut = make_dataset(64, 32, 0.02, 0.5, seed=0, amplitude=1.0)
    torch.manual_seed(0)
    model = build_fno(modes=8, width=16)
    before = evaluate_model(model, u0, ut)
    train_operator(model, u0, ut, epochs=15, seed=0)
    assert evaluate_model(model, u0, ut) < 0.8 * before
    # same weights evaluate on a finer grid without error
    u0_hi, ut_hi = make_dataset(4, 64, 0.02, 0.5, seed=1, amplitude=1.0)
    assert np.isfinite(evaluate_model(model, u0_hi, ut_hi))


def test_mlp_is_tied_to_its_grid() -> None:
    mlp = MLPOperator(32, hidden=16, layers=2)
    assert mlp(torch.zeros(2, 32)).shape == (2, 32)
    try:
        mlp(torch.zeros(2, 64))
    except RuntimeError:
        return
    raise AssertionError("MLP should reject a different grid size")


def test_identity_baseline_and_param_count() -> None:
    u = np.ones((2, 8))
    assert identity_error(u, u) == 0.0
    assert count_params(MLPOperator(8, hidden=4, layers=1)) == 8 * 4 + 4 + 4 * 8 + 8
