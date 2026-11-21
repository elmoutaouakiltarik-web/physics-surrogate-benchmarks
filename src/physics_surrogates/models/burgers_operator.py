"""Operator learning on the *nonlinear* periodic viscous Burgers equation.

Task: learn G: u(., 0) -> u(., T) for u_t + u u_x = nu u_xx on [0, 1), periodic.
Unlike the heat equation there is no closed form, so the reference is the numerical
pseudo-spectral solver in ``physics_surrogates.solvers`` (validated in tests/test_solvers.py);
errors reported here are model error plus the (much smaller) reference-solver error.

Models compared at equal training data:
  * ``FNO1d``  - resolution-independent, evaluated zero-shot on finer grids;
  * ``MLPOperator`` - dense baseline of comparable parameter count, tied to the training grid;
  * identity - predicts u(T) = u(0); measures how much the dynamics actually change the field.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from physics_surrogates.models.fno import FNO1d, random_periodic_fields, relative_l2
from physics_surrogates.solvers import burgers_solve


class MLPOperator(nn.Module):
    """Dense u0 -> u(T) map on a fixed grid (no resolution independence)."""

    def __init__(self, n_grid: int, hidden: int = 256, layers: int = 3) -> None:
        super().__init__()
        dims = [n_grid] + [hidden] * layers
        blocks: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*blocks, nn.Linear(hidden, n_grid))
        self.n_grid = n_grid

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


def make_dataset(
    n_samples: int, n_grid: int, nu: float, t: float, seed: int, amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Random smooth ICs and their Burgers solutions at time ``t`` (reference solver)."""
    u0 = amplitude * random_periodic_fields(n_samples, n_grid, seed=seed)
    return u0, burgers_solve(u0, nu, t)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_operator(
    model: nn.Module,
    u0: np.ndarray,
    ut: np.ndarray,
    epochs: int = 200,
    seed: int = 0,
    lr: float = 3e-3,
    batch: int = 64,
) -> list[float]:
    """Train with relative-L2 loss, Adam + cosine schedule. Returns per-epoch mean loss."""
    torch.manual_seed(seed)
    x = torch.tensor(u0, dtype=torch.float32)
    y = torch.tensor(ut, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: list[float] = []
    for _ in range(epochs):
        losses = []
        for idx in torch.randperm(len(x)).split(batch):
            opt.zero_grad()
            loss = (torch.norm(model(x[idx]) - y[idx], dim=-1) / torch.norm(y[idx], dim=-1)).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        history.append(float(np.mean(losses)))
    return history


@torch.no_grad()
def evaluate_model(model: nn.Module, u0: np.ndarray, ut: np.ndarray) -> float:
    model.eval()
    return relative_l2(model(torch.tensor(u0, dtype=torch.float32)).numpy(), ut)


def identity_error(u0: np.ndarray, ut: np.ndarray) -> float:
    return relative_l2(u0, ut)


def build_fno(modes: int = 16, width: int = 32) -> FNO1d:
    return FNO1d(modes=modes, width=width)
