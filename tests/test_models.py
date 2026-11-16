"""
Tests for surrogate models (PINN, FNO, DeepONet).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_surrogates.models.pinns import BurgersPINN, PINNNet, SinActivation


class TestPINNNet:
    """Test PINNNet backbone."""

    def test_initialization(self):
        """Test network initialization."""
        net = PINNNet([2, 64, 64, 1], activation="tanh")
        assert net is not None

    def test_forward_pass(self):
        """Test forward pass through network."""
        net = PINNNet([2, 64, 64, 1], activation="tanh")
        x = torch.randn(10, 2)
        y = net(x)
        assert y.shape == (10, 1)

    def test_activation_functions(self):
        """Test different activation functions."""
        for act in ["tanh", "sin", "gelu", "swish"]:
            net = PINNNet([2, 32, 1], activation=act)
            x = torch.randn(5, 2)
            y = net(x)
            assert y.shape == (5, 1)

    def test_sin_activation(self):
        """Test custom sin activation."""
        act = SinActivation()
        x = torch.tensor([[0.0], [np.pi/2], [np.pi]])
        y = act(x)
        assert torch.allclose(y[0], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(y[1], torch.tensor(1.0), atol=1e-5)


class TestBurgersPINN:
    """Test Burgers PINN implementation."""

    def test_initialization(self):
        """Test PINN initialization."""
        pinn = BurgersPINN(nu=0.01, device="cpu")
        assert pinn is not None
        assert pinn.nu == 0.01

    def test_residual_computation(self):
        """Test residual computation."""
        pinn = BurgersPINN(device="cpu")
        x = torch.randn(10, 1)
        t = torch.randn(10, 1)
        res = pinn.residual(x, t)
        assert res.shape == (10, 1)

    def test_prediction(self):
        """Test prediction on data."""
        pinn = BurgersPINN(device="cpu")
        x = np.linspace(-1, 1, 50)
        u = pinn.predict(x, t=0.5)
        assert u.shape == (50,)

    def test_train_short(self):
        """Test quick training (smoke test)."""
        pinn = BurgersPINN(device="cpu")
        metrics = pinn.train(
            n_colloc=100,
            n_ic=50,
            n_bc=50,
            epochs_adam=10,
            epochs_lbfgs=1,
            use_rar=False,
        )
        assert "final_loss" in metrics
        assert "elapsed_time" in metrics
        assert metrics["final_loss"] > 0

    def test_save_load_weights(self, tmp_path):
        """Test weight saving and loading."""
        pinn1 = BurgersPINN(device="cpu")

        # Save
        weight_file = tmp_path / "test_pinn.pt"
        pinn1.save_weights(weight_file)
        assert weight_file.exists()

        # Load
        pinn2 = BurgersPINN(device="cpu")
        pinn2.load_weights(weight_file)

        # Verify same output
        x = np.linspace(-1, 1, 10)
        u1 = pinn1.predict(x, t=0.5)
        u2 = pinn2.predict(x, t=0.5)
        np.testing.assert_allclose(u1, u2, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
