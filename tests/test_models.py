"""Tests for the PINN implementation."""

import numpy as np
import pytest
import torch

from physics_surrogates.models.pinns import BurgersPINN, PINNNet, RARSampler, SinActivation


class TestPINNNet:
    def test_initialization(self):
        net = PINNNet([2, 64, 64, 1], activation="tanh")
        assert isinstance(net, torch.nn.Module)

    def test_forward_shape(self):
        net = PINNNet([2, 64, 64, 1], activation="tanh")
        assert net(torch.randn(10, 2)).shape == (10, 1)

    @pytest.mark.parametrize("activation", ["tanh", "sin", "gelu", "swish"])
    def test_activations(self, activation):
        net = PINNNet([2, 32, 1], activation=activation)
        assert net(torch.randn(5, 2)).shape == (5, 1)

    def test_unknown_activation_falls_back_to_tanh(self):
        net = PINNNet([2, 4, 1], activation="not-a-real-activation")
        assert net(torch.randn(3, 2)).shape == (3, 1)

    def test_sin_activation(self):
        act = SinActivation()
        y = act(torch.tensor([[0.0], [np.pi / 2], [np.pi]]))
        assert torch.allclose(y[0], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(y[1], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(y[2], torch.tensor(0.0), atol=1e-6)


class TestBurgersPINN:
    def test_initialization(self):
        pinn = BurgersPINN(nu=0.01, device="cpu")
        assert pinn.nu == 0.01
        assert sum(p.numel() for p in pinn.net.parameters()) == 12737

    def test_residual_shape(self):
        pinn = BurgersPINN(device="cpu")
        assert pinn.residual(torch.randn(10, 1), torch.randn(10, 1)).shape == (10, 1)

    def test_prediction(self):
        pinn = BurgersPINN(device="cpu")
        assert pinn.predict(np.linspace(-1, 1, 50), t=0.5).shape == (50,)

    def test_train_reports_the_configuration_it_used(self):
        """The result dict is what feeds the JSON, so it must describe the run."""
        pinn = BurgersPINN(device="cpu")
        metrics = pinn.train(
            n_colloc=100,
            n_ic=30,
            n_bc=40,
            epochs_adam=3,
            epochs_lbfgs=1,
            use_rar=False,
        )
        assert metrics["n_ic"] == 30
        assert metrics["n_bc"] == 40
        assert metrics["n_colloc"] == 100
        assert metrics["n_colloc_final"] == 100
        assert metrics["n_params"] == 12737
        assert metrics["final_loss"] > 0
        assert metrics["elapsed_time"] > 0

    def test_rar_changes_the_collocation_count(self):
        pinn = BurgersPINN(device="cpu")
        metrics = pinn.train(
            n_colloc=200,
            n_ic=20,
            n_bc=20,
            epochs_adam=4,
            epochs_lbfgs=0,
            use_rar=True,
            rar_interval=2,
        )
        assert metrics["n_colloc_final"] > 200

    def test_save_and_load_weights(self, tmp_path):
        pinn1 = BurgersPINN(device="cpu")
        path = tmp_path / "pinn.pt"
        pinn1.save_weights(path)
        assert path.exists()

        pinn2 = BurgersPINN(device="cpu")
        pinn2.load_weights(path)
        x = np.linspace(-1, 1, 10)
        np.testing.assert_allclose(pinn1.predict(x, 0.5), pinn2.predict(x, 0.5), rtol=1e-5)


class TestRARSampler:
    def test_returns_requested_number_of_points_inside_the_domain(self):
        sampler = RARSampler(n_add=50, n_cand=500, top_frac=0.1)

        def residual(x, t):
            return torch.sin(torch.pi * x) * torch.cos(torch.pi * t)

        x_new, t_new = sampler.sample(
            residual, domain_x=(-1.0, 1.0), domain_t=(0.0, 1.0), device=torch.device("cpu")
        )
        assert x_new.shape == (50, 1)
        assert t_new.shape == (50, 1)
        assert float(x_new.min()) >= -1.0 and float(x_new.max()) <= 1.0
        assert float(t_new.min()) >= 0.0 and float(t_new.max()) <= 1.0
        assert not x_new.requires_grad
