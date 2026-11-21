"""Closed-form reference solutions used to score the surrogates.

Every benchmark in this repository is compared against an analytical solution, so
the reported error is the error of the model and not the error of a numerical
solver. ``cole_hopf_burgers`` is the only non-trivial one and it needs care: at
``nu = 0.01/pi`` the Cole-Hopf integrand spans roughly ``exp(+-50)``, so it is
evaluated in log space and shifted by its maximum (log-sum-exp) before
exponentiation. A naive evaluation overflows.
"""

from __future__ import annotations

import math

import numpy as np

# numpy >= 2.0 renamed trapz -> trapezoid.
_trapz = getattr(np, "trapezoid", np.trapz)


def burgers_initial_condition(x: np.ndarray | float) -> np.ndarray | float:
    """u(x, 0) = -sin(pi x) on the benchmark domain x in [-1, 1]."""
    return -np.sin(math.pi * np.asarray(x, dtype=np.float64))


def cole_hopf_burgers(
    x: np.ndarray | float,
    t: float,
    nu: float,
    eta_half_width: float = 4.0,
    n_eta: int = 160001,
) -> np.ndarray | float:
    """Reference solution of viscous Burgers via the Cole-Hopf integral.

    PDE:  u_t + u*u_x = nu*u_xx,  u(x,0) = -sin(pi*x),  u(+-1,t) = 0.

    The Cole-Hopf transform turns the problem into the heat equation, whose
    solution can be written in closed form (Basdevant et al., 1986):

        u(x,t) = - I_num / I_den
        I_num  = int sin(pi*(x-eta)) * f(x-eta) * exp(-eta^2/(4*nu*t)) d(eta)
        I_den  = int             f(x-eta) * exp(-eta^2/(4*nu*t)) d(eta)
        f(y)   = exp(-cos(pi*y) / (2*pi*nu))

    Args:
        x: Scalar or array of spatial coordinates in [-1, 1].
        t: Time (>= 0).
        nu: Kinematic viscosity.
        eta_half_width: Truncation of the eta integral. The Gaussian factor has
            standard deviation ``sqrt(2*nu*t)`` (~0.06 at nu=0.01/pi, t=0.5), so
            4.0 is hundreds of standard deviations and the truncation error is far
            below float64 resolution.
        n_eta: Number of quadrature nodes (composite trapezoid).

    Returns:
        ``np.ndarray`` with the shape of ``x``, or ``float`` for scalar input.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if t < 1e-12:
        out = burgers_initial_condition(x_arr)
        return out if np.ndim(x) else float(out[0])

    eta = np.linspace(-eta_half_width, eta_half_width, n_eta)
    d_eta = eta[1] - eta[0]
    gauss_log = -(eta**2) / (4.0 * nu * t)

    u = np.empty_like(x_arr)
    for i, xi in enumerate(x_arr):
        y = xi - eta
        # log of the full integrand weight, shifted by its max for stability
        w = -np.cos(np.pi * y) / (2.0 * np.pi * nu) + gauss_log
        w -= w.max()
        weight = np.exp(w)
        num = _trapz(np.sin(np.pi * y) * weight, dx=d_eta)
        den = _trapz(weight, dx=d_eta)
        u[i] = -num / den

    return u if np.ndim(x) else float(u[0])


def burgers_spectral_reference(
    t_final: float,
    nu: float,
    n_x: int = 256,
    n_steps: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent numerical solution of the same Burgers problem.

    Uses a Fourier pseudo-spectral discretisation on the periodic extension of
    ``[-1, 1)`` (valid because the exact solution is odd in x and vanishes at
    ``x = +-1``) with explicit RK4 in time. This is deliberately a *different*
    method from :func:`cole_hopf_burgers` so the two can be cross-checked; see
    ``tests/test_reference_solutions.py``.

    Returns:
        ``(x, u)`` with ``x`` on ``[-1, 1)`` and ``u`` at ``t = t_final``.
    """
    x = np.linspace(-1.0, 1.0, n_x, endpoint=False)
    length = 2.0
    k = 2.0 * np.pi * np.fft.fftfreq(n_x, d=length / n_x)

    # 2/3 dealiasing keeps the quadratic nonlinearity from folding high modes back.
    dealias = np.abs(k) <= (n_x // 3) * (2.0 * np.pi / length)
    dt = t_final / n_steps

    def rhs(u: np.ndarray) -> np.ndarray:
        u_hat = np.fft.fft(u)
        u_x = np.fft.ifft(1j * k * u_hat).real
        u_xx = np.fft.ifft(-(k**2) * u_hat).real
        out_hat = np.fft.fft(-u * u_x + nu * u_xx)
        return np.fft.ifft(out_hat * dealias).real

    u = burgers_initial_condition(x)
    for _ in range(n_steps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x, u
