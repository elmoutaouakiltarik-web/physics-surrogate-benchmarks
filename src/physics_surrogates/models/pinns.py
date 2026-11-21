"""
Physics-Informed Neural Networks (PINNs) for classical CFD benchmarks.

Implements:
  - PINNNet: fully connected backbone with sin/tanh/gelu/swish activations
  - BurgersPINN: 1D viscous Burgers equation (u_t + u*u_x = nu*u_xx)
  - RARSampler: Residual Adaptive Refinement for efficient training

Only the 1D Burgers solver is implemented; the 2D Navier-Stokes PINNs are not
part of this release, and nothing in the repository claims otherwise.

Training pipeline:
  1. Adam (library default 5000 epochs) - initial rapid convergence
  2. L-BFGS (library default 500 iterations) - fine-tuning to the minimum
  3. RAR (optional) - add candidate points where the residual is largest

The published benchmark uses 10,000 Adam epochs followed by 3,000 L-BFGS
iterations with RAR every 2,000 epochs; see ``scripts/run_benchmark_real.py``.

All residuals are computed with automatic differentiation (torch.autograd.grad),
so there are no finite-difference approximations in the loss.

References:
  Raissi et al. "Physics-informed neural networks", JCP 2019
  Lu et al. "DeepXDE", SIAM Review 2021
  Wu et al. "A comprehensive study of non-adaptive and RAR", arXiv 2022
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Backbone Network
# ══════════════════════════════════════════════════════════════════════════════

class SinActivation(nn.Module):
    """Sine activation function for PINN networks."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class PINNNet(nn.Module):
    """
    Fully connected network with learnable activations (sin/tanh/gelu).
    Architecture: [input] -> N x [Linear -> Act] -> [output]
    Initialization: Xavier normal (optimal for Tanh/Sin).
    """
    def __init__(self, layers: list[int], activation: str = "tanh"):
        super().__init__()
        act_map = {
            "tanh": nn.Tanh,
            "sin": SinActivation,
            "gelu": nn.GELU,
            "swish": lambda: nn.SiLU(),
        }
        act_fn = act_map.get(activation, nn.Tanh)
        seqs: list[nn.Module] = []
        for i in range(len(layers) - 1):
            seqs.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                seqs.append(act_fn())
        self.net = nn.Sequential(*seqs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute dy/dx via autograd."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


# ══════════════════════════════════════════════════════════════════════════════
# Residual Adaptive Refinement (RAR)
# ══════════════════════════════════════════════════════════════════════════════

class RARSampler:
    """
    Residual Adaptive Refinement (Wu et al. 2022).
    Adds collocation points in high-residual regions during training.
    Works with any PINN exposing a residual() method.
    """

    def __init__(
        self,
        n_add: int = 1000,  # points to add per RAR iteration
        n_cand: int = 50000,  # candidates to evaluate
        top_frac: float = 0.10,  # fraction of top candidates to keep
    ):
        self.n_add = n_add
        self.n_cand = n_cand
        self.top_frac = top_frac

    def sample(
        self,
        residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        domain_x: tuple[float, float],
        domain_t: tuple[float, float],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample new points where residual is maximum.
        residual_fn(x, t) -> scalar residual for each point
        Returns: (x_new, t_new) tensors of shape [n_add, 1]
        """
        x_cand = (
            torch.rand(self.n_cand, 1, device=device) *
            (domain_x[1] - domain_x[0]) + domain_x[0]
        )
        t_cand = (
            torch.rand(self.n_cand, 1, device=device) *
            (domain_t[1] - domain_t[0]) + domain_t[0]
        )
        x_cand.requires_grad_(True)
        t_cand.requires_grad_(True)

        with torch.enable_grad():
            res = residual_fn(x_cand, t_cand)
            res_val = res.detach().abs().squeeze()

        # Select top_frac points with highest residual
        n_top = max(self.n_add, int(self.n_cand * self.top_frac))
        _, top_idx = torch.topk(res_val, min(n_top, len(res_val)))
        top_idx = top_idx[: self.n_add]

        x_new = x_cand[top_idx].detach().requires_grad_(False)
        t_new = t_cand[top_idx].detach().requires_grad_(False)
        return x_new, t_new


# ══════════════════════════════════════════════════════════════════════════════
# Burgers PINN
# ══════════════════════════════════════════════════════════════════════════════

class BurgersPINN:
    """
    Physics-Informed Neural Network for 1D viscous Burgers equation:
        u_t + u * u_x = nu * u_xx

    Domain: x in [-1, 1], t in [0, 1]
    IC: u(x, 0) = -sin(pi*x)
    BC: u(-1, t) = u(1, t) = 0
    """

    def __init__(
        self,
        nu: float = 0.01 / math.pi,
        layers: list[int] | None = None,
        activation: str = "tanh",
        device: str = "cpu",
    ):
        self.nu = nu
        self.device = torch.device(device)
        layers = layers or [2, 64, 64, 64, 64, 1]
        self.net = PINNNet(layers, activation).to(self.device)
        self.losses: list[float] = []

    def _u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))

    def residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        u = self._u(x, t)
        u_t = _grad(u, t)
        u_x = _grad(u, x)
        u_xx = _grad(u_x, x)
        return u_t + u * u_x - self.nu * u_xx

    def _ic_residual(self, x: torch.Tensor) -> torch.Tensor:
        t0 = torch.zeros_like(x)
        u_pred = self._u(x, t0)
        return u_pred - (-torch.sin(math.pi * x))

    def _bc_residual(self, t: torch.Tensor) -> torch.Tensor:
        x_l = -torch.ones_like(t)
        x_r = torch.ones_like(t)
        return torch.cat([self._u(x_l, t), self._u(x_r, t)])

    def train(
        self,
        n_colloc: int = 10000,
        n_ic: int = 200,
        n_bc: int = 200,
        epochs_adam: int = 5000,
        epochs_lbfgs: int = 500,
        lr: float = 1e-3,
        use_rar: bool = False,
        rar_interval: int = 1000,
    ) -> dict:
        """
        Training loop: Adam + L-BFGS with optional RAR.
        """
        x_c = (torch.rand(n_colloc, 1) * 2 - 1).to(self.device)
        t_c = torch.rand(n_colloc, 1).to(self.device)
        x_ic = (torch.rand(n_ic, 1) * 2 - 1).to(self.device)
        t_bc = torch.rand(n_bc, 1).to(self.device)

        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

        rar = RARSampler() if use_rar else None

        start_time = time.time()
        for epoch in range(epochs_adam):
            optimizer.zero_grad()
            loss_f = (
                self.residual(
                    x_c.clone().requires_grad_(True),
                    t_c.clone().requires_grad_(True),
                )
                .pow(2)
                .mean()
            )
            loss_ic = self._ic_residual(x_ic).pow(2).mean()
            loss_bc = self._bc_residual(t_bc).pow(2).mean()
            loss = loss_f + 10 * loss_ic + 5 * loss_bc
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 500 == 0:
                self.losses.append(loss.item())
                logger.info(
                    f"  Burgers PINN epoch {epoch}: total={loss.item():.4e} "
                    f"(f={loss_f.item():.2e} ic={loss_ic.item():.2e} bc={loss_bc.item():.2e})"
                )

            # RAR: add high-residual points
            if use_rar and rar is not None and epoch > 0 and epoch % rar_interval == 0:
                x_new, t_new = rar.sample(
                    lambda x, t: self.residual(x, t),
                    domain_x=(-1.0, 1.0),
                    domain_t=(0.0, 1.0),
                    device=self.device,
                )
                x_c = torch.cat([x_c, x_new], dim=0)
                t_c = torch.cat([t_c, t_new], dim=0)
                logger.info(f"  RAR: added {len(x_new)} points, total={len(x_c)}")

        # L-BFGS fine-tuning
        lbfgs = optim.LBFGS(
            self.net.parameters(),
            lr=0.1,
            max_iter=epochs_lbfgs,
            history_size=100,
            tolerance_grad=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            lf = (
                self.residual(
                    x_c.clone().requires_grad_(True),
                    t_c.clone().requires_grad_(True),
                )
                .pow(2)
                .mean()
            )
            li = self._ic_residual(x_ic).pow(2).mean()
            lb = self._bc_residual(t_bc).pow(2).mean()
            loss = lf + 10 * li + 5 * lb
            loss.backward()
            return loss

        if epochs_lbfgs > 0:
            lbfgs.step(closure)
            final_loss = closure().item()
            self.losses.append(final_loss)
        else:
            final_loss = closure().item()
        elapsed = time.time() - start_time
        logger.info(f"  Burgers PINN complete. Final loss: {final_loss:.4e} [{elapsed:.1f}s]")
        # Every value reported in the committed JSON comes from here; nothing is
        # re-typed by hand, so the record cannot drift from the run.
        return {
            "final_loss": final_loss,
            "nu": self.nu,
            "n_colloc": n_colloc,
            "n_colloc_final": len(x_c),
            "n_ic": n_ic,
            "n_bc": n_bc,
            "epochs_adam": epochs_adam,
            "epochs_lbfgs": epochs_lbfgs,
            "n_params": sum(p.numel() for p in self.net.parameters()),
            "elapsed_time": elapsed,
            "model_type": "BurgersPINN",
        }

    def predict(self, x: np.ndarray, t: float) -> np.ndarray:
        """Predict solution at time t."""
        self.net.eval()
        with torch.no_grad():
            xv = torch.tensor(
                x.reshape(-1, 1), dtype=torch.float32, device=self.device
            )
            tv = torch.full_like(xv, t)
            return self.net(torch.cat([xv, tv], dim=1)).cpu().numpy().flatten()

    def save_weights(self, path: str | Path):
        """Save model weights."""
        torch.save(self.net.state_dict(), path)
        logger.info(f"Saved PINN weights to {path}")

    def load_weights(self, path: str | Path):
        """Load model weights (state dict only; no pickled objects are executed)."""
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        logger.info(f"Loaded PINN weights from {path}")
