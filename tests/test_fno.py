"""Tests for the 1D Fourier Neural Operator."""

import numpy as np
import torch

from physics_surrogates.models.fno import (
    FNO1d,
    SpectralConv1d,
    evaluate,
    heat_solution,
    random_periodic_fields,
    relative_l2,
    train_fno_heat,
)


def test_shapes_any_resolution():
    model = FNO1d(modes=8, width=16)
    for n in (32, 64, 257):
        assert model(torch.randn(3, n)).shape == (3, n)


def test_spectral_conv_ignores_high_modes():
    conv = SpectralConv1d(1, 1, modes=4)
    x = torch.linspace(0, 1, 65)[:-1]
    high = torch.sin(2 * torch.pi * 20 * x).reshape(1, 1, -1)
    assert torch.allclose(conv(high), torch.zeros_like(high), atol=1e-5)


def test_heat_solution_decays_each_mode_exactly():
    n, nu, t = 128, 0.01, 0.5
    x = np.arange(n) / n
    u0 = np.sin(2 * np.pi * 3 * x)
    expected = np.exp(-nu * (2 * np.pi * 3) ** 2 * t) * u0
    np.testing.assert_allclose(heat_solution(u0, nu, t), expected, atol=1e-12)


def test_relative_l2_zero_for_exact():
    u = random_periodic_fields(4, 32, seed=1)
    assert relative_l2(u, u) == 0.0


def test_learns_heat_operator_and_transfers_to_finer_grid():
    nu, t = 0.01, 1.0
    model, history = train_fno_heat(n_train=256, n_grid=64, nu=nu, t=t, epochs=60, seed=0)
    assert history[-1] < history[0]
    err_same = evaluate(model, random_periodic_fields(64, 64, seed=99), nu, t)
    err_fine = evaluate(model, random_periodic_fields(64, 256, seed=99), nu, t)
    assert err_same < 0.05, err_same
    # Zero-shot super-resolution: trained at n=64, evaluated at n=256.
    assert err_fine < 0.10, err_fine
