"""Reference solver for the periodic viscous Burgers equation.

    u_t + u u_x = nu u_xx   on [0, 1), periodic.

Pseudo-spectral in space (2/3-rule dealiasing) and integrating-factor RK4 in time: the
stiff linear term exp(-nu k^2 t) is integrated exactly, the nonlinear term with RK4.
Unlike the heat equation this problem has no closed form, so the reference is a *numerical*
solution; ``tests/test_solvers.py`` checks it by (i) temporal convergence, (ii) agreement
with the exact heat solution in the linear limit, (iii) conservation of the mean.
"""

from __future__ import annotations

import math

import numpy as np


def burgers_solve(u0: np.ndarray, nu: float, t_final: float, dt: float = 1e-3) -> np.ndarray:
    """Advance ``u0`` (shape ``(..., n)``, uniform periodic grid on [0, 1)) to ``t_final``.

    Args:
        u0: Initial fields; the last axis is space.
        nu: Viscosity (> 0).
        t_final: Final time (> 0).
        dt: Time step; the last step is shortened so that ``t_final`` is hit exactly.

    Returns:
        Array shaped like ``u0``.
    """
    if nu <= 0 or t_final <= 0 or dt <= 0:
        raise ValueError("nu, t_final and dt must be positive")
    n = u0.shape[-1]
    k = 2 * math.pi * np.fft.rfftfreq(n, d=1.0 / n)  # angular wavenumbers
    mask = (np.abs(np.fft.rfftfreq(n, d=1.0 / n)) <= n / 3).astype(float)  # 2/3 rule

    def nonlinear(v: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(v, n=n)
        return -0.5j * k * np.fft.rfft(u * u) * mask

    v = np.fft.rfft(u0) * mask
    steps = max(1, math.ceil(t_final / dt - 1e-12))
    h = t_final / steps  # equal steps that land exactly on t_final
    e = np.exp(-nu * k**2 * h / 2)
    e2 = e * e
    for _ in range(steps):
        k1 = nonlinear(v)
        k2 = nonlinear(e * (v + 0.5 * h * k1))
        k3 = nonlinear(e * v + 0.5 * h * k2)
        k4 = nonlinear(e2 * v + h * e * k3)
        v = e2 * v + h / 6 * (e2 * k1 + 2 * e * (k2 + k3) + k4)
    return np.fft.irfft(v, n=n)
