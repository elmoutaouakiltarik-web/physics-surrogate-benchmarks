"""1D Fourier Neural Operator (Li et al., 2021, arXiv:2010.08895).

A compact, dependency-free (PyTorch only) implementation of FNO1d plus an exact
operator-learning task to check it against: the periodic heat equation

    u_t = nu * u_xx  on [0, 1), periodic,

whose solution operator G: u(., 0) -> u(., T) is diagonal in Fourier space,
u_hat_k(T) = exp(-nu * (2 pi k)^2 * T) * u_hat_k(0). That gives a ground truth with
no numerical solver error, so the reported error is the model's error only.

This task is deliberately easy: the operator is linear and diagonal in the basis
the spectral layer uses. It verifies the implementation and the resolution
invariance of the architecture; it is not evidence about nonlinear flow.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_MODES = 16
DEFAULT_WIDTH = 32
DEFAULT_LAYERS = 4


class SpectralConv1d(nn.Module):
    """Multiply the lowest ``modes`` Fourier coefficients by learned complex weights."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, channels, n)
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0], self.weight.shape[1], x_ft.shape[-1], dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=n)


class FNO1d(nn.Module):
    """FNO for 1D fields: lift -> L x (spectral conv + pointwise conv, GELU) -> project.

    Input  (batch, n) field sampled on a uniform grid of any size n.
    Output (batch, n). Because the weights live in Fourier space, the same trained
    model can be evaluated on a finer or coarser grid than the one it was trained on.
    """

    def __init__(
        self,
        modes: int = DEFAULT_MODES,
        width: int = DEFAULT_WIDTH,
        layers: int = DEFAULT_LAYERS,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        self.layers = layers
        self.lift = nn.Linear(2, width)  # (u, x) -> width
        self.spectral = nn.ModuleList(SpectralConv1d(width, width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        n = u.shape[-1]
        grid = torch.linspace(0, 1, n + 1, device=u.device)[:-1].expand(u.shape[0], n)
        h = self.lift(torch.stack([u, grid], dim=-1)).permute(0, 2, 1)  # (b, width, n)
        for i, (spec, pw) in enumerate(zip(self.spectral, self.pointwise, strict=True)):
            h = spec(h) + pw(h)
            if i < len(self.spectral) - 1:
                h = nn.functional.gelu(h)
        return self.project(h.permute(0, 2, 1)).squeeze(-1)


def random_periodic_fields(
    n_samples: int, n_grid: int, n_modes: int = 8, seed: int = 0
) -> np.ndarray:
    """Smooth random periodic initial conditions with decaying Fourier spectrum."""
    rng = np.random.default_rng(seed)
    x = np.arange(n_grid) / n_grid
    k = np.arange(1, n_modes + 1)
    a = rng.normal(size=(n_samples, n_modes)) / k**1.5
    b = rng.normal(size=(n_samples, n_modes)) / k**1.5
    phase = 2 * math.pi * np.outer(k, x)  # (modes, n)
    return a @ np.cos(phase) + b @ np.sin(phase)


def heat_solution(u0: np.ndarray, nu: float, t: float) -> np.ndarray:
    """Exact periodic heat-equation solution on [0, 1) via FFT."""
    n = u0.shape[-1]
    k = np.fft.rfftfreq(n, d=1.0 / n)
    return np.fft.irfft(np.fft.rfft(u0) * np.exp(-nu * (2 * math.pi * k) ** 2 * t), n=n)


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean over samples of ||pred - true||_2 / ||true||_2."""
    num = np.linalg.norm(pred - true, axis=-1)
    den = np.linalg.norm(true, axis=-1)
    return float(np.mean(num / den))


def train_fno_heat(
    n_train: int = 512,
    n_grid: int = 64,
    nu: float = 0.01,
    t: float = 1.0,
    epochs: int = 300,
    seed: int = 0,
    modes: int = DEFAULT_MODES,
    width: int = DEFAULT_WIDTH,
    batch_size: int = 64,
) -> tuple[FNO1d, list[float]]:
    """Train FNO1d to learn u0 -> u(T) for the heat equation.

    Returns the model and the history of the *epoch-mean* relative-L2 loss (the
    full training set is averaged every epoch; scoring a single mini-batch makes
    the reported loss depend on how the permutation happened to end).
    """
    torch.manual_seed(seed)
    u0 = random_periodic_fields(n_train, n_grid, seed=seed)
    ut = heat_solution(u0, nu, t)
    x = torch.tensor(u0, dtype=torch.float32)
    y = torch.tensor(ut, dtype=torch.float32)
    model = FNO1d(modes=modes, width=width)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for idx in torch.randperm(n_train).split(batch_size):
            opt.zero_grad()
            pred = model(x[idx])
            loss = (torch.norm(pred - y[idx], dim=-1) / torch.norm(y[idx], dim=-1)).mean()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        sched.step()
        history.append(epoch_loss / n_train)
    return model, history


@torch.no_grad()
def evaluate(model: FNO1d, u0: np.ndarray, nu: float, t: float) -> float:
    """Relative L2 of the predicted u(., T) against the exact Fourier solution."""
    model.eval()
    pred = model(torch.tensor(u0, dtype=torch.float32)).numpy()
    return relative_l2(pred, heat_solution(u0, nu, t))


@torch.no_grad()
def predict(model: FNO1d, u0: np.ndarray) -> np.ndarray:
    """Evaluate the learned operator on a batch of initial conditions."""
    model.eval()
    return model(torch.tensor(u0, dtype=torch.float32)).numpy()


def save_fno_weights(model: FNO1d, path: str | Path) -> None:
    """Save the state dict together with the architecture it belongs to."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "modes": model.modes,
            "width": model.width,
            "layers": model.layers,
        },
        path,
    )


def load_fno_weights(path: str | Path) -> FNO1d:
    """Rebuild an FNO1d from a file written by :func:`save_fno_weights`."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = FNO1d(modes=blob["modes"], width=blob["width"], layers=blob["layers"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
