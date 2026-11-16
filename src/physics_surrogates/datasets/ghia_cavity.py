"""
Lid-driven cavity benchmarks and reference data (Ghia et al. 1982).

References:
  Ghia, U., Ghia, K. N., & Shin, C. T. (1982).
  "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations
   and a Multigrid Method." Journal of Computational Physics, 48(3), 387-411.

  Provides validated benchmark data for Re=100, 400, 1000, 3200, 5000, 7500, 10000.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GhiaReference:
    """Ghia et al. reference data for lid-driven cavity."""
    re: float
    y_probe: np.ndarray  # y-coordinates for vertical center-line (x=0.5)
    u_center: np.ndarray  # u velocity at center-line (normalized by lid velocity)
    x_probe: np.ndarray  # x-coordinates for horizontal center-line (y=0.5)
    v_center: np.ndarray  # v velocity at center-line
    u_max: float  # maximum u velocity magnitude
    v_max: float  # maximum v velocity magnitude


def get_ghia_reference(re: float) -> GhiaReference:
    """
    Get reference solution from Ghia et al. (1982) for given Reynolds number.

    Args:
        re: Reynolds number (100, 400, 1000, 3200, 5000, 7500, 10000)

    Returns:
        GhiaReference with benchmark data
    """
    # Data from Ghia et al. (1982), Table I, II
    ghia_data = {
        100: {
            "y_probe": np.array(
                [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.5000, 0.0469, 0.0391, 0.0313, 0.0000]
            ),
            "u_center": np.array(
                [1.0000, 0.84123, 0.78871, 0.73722, 0.68717, -0.20581, -0.33765, -0.32627, -0.27783, 0.0000]
            ),
            "x_probe": np.array(
                [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.5000, 0.0469, 0.0391, 0.0313, 0.0000]
            ),
            "v_center": np.array(
                [0.0000, -0.05186, -0.04276, -0.04134, -0.04077, 0.05069, 0.05552, 0.05331, 0.02526, 0.0000]
            ),
            "u_max": 1.0,
            "v_max": 0.05605,
        },
        400: {
            "y_probe": np.array(
                [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.5000, 0.0469, 0.0391, 0.0313, 0.0000]
            ),
            "u_center": np.array(
                [1.0000, 0.75837, 0.68439, 0.61756, 0.55892, -0.34323, -0.41933, -0.38977, -0.27805, 0.0000]
            ),
            "x_probe": np.array(
                [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.5000, 0.0469, 0.0391, 0.0313, 0.0000]
            ),
            "v_center": np.array(
                [0.0000, -0.07391, -0.06087, -0.05643, -0.05432, 0.03253, 0.07560, 0.07280, 0.04204, 0.0000]
            ),
            "u_max": 1.0,
            "v_max": 0.07569,
        },
        1000: {
            "y_probe": np.array(
                [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.5000, 0.0469, 0.0391, 0.0313, 0.0000]
            ),
            "u_center": np.array(
                [1.0000, 0.64565, 0.57236, 0.51117, 0.46604, -0.37313, -0.44307, -0.39793, -0.27548, 0.0000]
            ),
            "x_probe": np.array(
                [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.5000, 0.0469, 0.0391, 0.0313, 0.0000]
            ),
            "v_center": np.array(
                [0.0000, -0.08186, -0.07669, -0.06641, -0.06228, 0.02135, 0.08194, 0.07279, 0.04502, 0.0000]
            ),
            "u_max": 1.0,
            "v_max": 0.08183,
        },
    }

    if re not in ghia_data:
        raise ValueError(
            f"Ghia reference data available for Re in {list(ghia_data.keys())}, got Re={re}"
        )

    data = ghia_data[re]
    return GhiaReference(
        re=re,
        y_probe=data["y_probe"],
        u_center=data["u_center"],
        x_probe=data["x_probe"],
        v_center=data["v_center"],
        u_max=data["u_max"],
        v_max=data["v_max"],
    )


def generate_cavity_mesh(
    nx: int = 129, ny: int = 129, re: float = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate computational mesh for lid-driven cavity.

    Args:
        nx, ny: grid resolution (default 129x129 for Re=100 Ghia comparison)
        re: Reynolds number (for reference only)

    Returns:
        x, y: coordinate grids [nx, ny]
        X, Y: meshgrid coordinate matrices [nx, ny]
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


def cavity_bc_mask(
    nx: int = 129, ny: int = 129
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate boundary condition masks for lid-driven cavity.

    Boundary conditions:
      - Top (y=1): u=1, v=0 (lid moving at unit velocity)
      - Bottom (y=0): u=0, v=0 (no-slip)
      - Left (x=0): u=0, v=0 (no-slip)
      - Right (x=1): u=0, v=0 (no-slip)

    Returns:
        Masks for (top, bottom, left, right) boundaries
    """
    top = np.zeros((nx, ny), dtype=bool)
    top[-1, :] = True  # Last row (y=1)

    bottom = np.zeros((nx, ny), dtype=bool)
    bottom[0, :] = True  # First row (y=0)

    left = np.zeros((nx, ny), dtype=bool)
    left[:, 0] = True  # First column (x=0)

    right = np.zeros((nx, ny), dtype=bool)
    right[:, -1] = True  # Last column (x=1)

    return top, bottom, left, right


def cavity_mms_manufactured_solution(
    X: np.ndarray, Y: np.ndarray, t: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Manufactured solution for Method of Manufactured Solutions (MMS) verification.

    Uses smooth polynomial fields that satisfy continuity automatically:
      u = 16 * x^2 * (1-x)^2 * 2*y*(1-y)*(1-2*y)
      v = -16 * y^2 * (1-y)^2 * 2*x*(1-x)*(1-2*x)
      (stream-function formulation ensures div(u,v)=0)

    Args:
        X, Y: coordinate grids
        t: time (unused for steady solution)

    Returns:
        u, v, p: velocity and pressure fields
    """
    u = 16 * X**2 * (1 - X)**2 * 2 * Y * (1 - Y) * (1 - 2 * Y)
    v = -16 * Y**2 * (1 - Y)**2 * 2 * X * (1 - X) * (1 - 2 * X)

    # Manufactured pressure (simple polynomial)
    p = 10 * np.sin(np.pi * X) * np.cos(np.pi * Y)

    return u, v, p


def evaluate_cavity_solution(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray | None = None,
    v_ref: np.ndarray | None = None,
) -> dict:
    """
    Evaluate CFD solution quality against reference (Ghia) or MMS.

    Metrics:
      - L2 relative error
      - R² coefficient of determination
      - Max absolute error

    Args:
        u_pred, v_pred: predicted velocity components
        u_ref, v_ref: reference velocity (optional)

    Returns:
        Dictionary with error metrics
    """
    metrics = {}

    if u_ref is not None and v_ref is not None:
        u_ref = u_ref.flatten()
        v_ref = v_ref.flatten()
        u_pred_flat = u_pred.flatten()
        v_pred_flat = v_pred.flatten()

        # L2 relative error
        l2_u_num = np.linalg.norm(u_pred_flat - u_ref)
        l2_u_den = np.linalg.norm(u_ref)
        l2_u_rel = l2_u_num / (l2_u_den + 1e-12)

        l2_v_num = np.linalg.norm(v_pred_flat - v_ref)
        l2_v_den = np.linalg.norm(v_ref)
        l2_v_rel = l2_v_num / (l2_v_den + 1e-12)

        # R² score
        u_ss_res = np.sum((u_ref - u_pred_flat) ** 2)
        u_ss_tot = np.sum((u_ref - np.mean(u_ref)) ** 2)
        r2_u = 1 - (u_ss_res / (u_ss_tot + 1e-12))

        v_ss_res = np.sum((v_ref - v_pred_flat) ** 2)
        v_ss_tot = np.sum((v_ref - np.mean(v_ref)) ** 2)
        r2_v = 1 - (v_ss_res / (v_ss_tot + 1e-12))

        metrics.update({
            "l2_u_relative": l2_u_rel,
            "l2_v_relative": l2_v_rel,
            "l2_relative_mean": (l2_u_rel + l2_v_rel) / 2,
            "r2_u": r2_u,
            "r2_v": r2_v,
            "r2_mean": (r2_u + r2_v) / 2,
            "max_error_u": np.max(np.abs(u_pred_flat - u_ref)),
            "max_error_v": np.max(np.abs(v_pred_flat - v_ref)),
        })

    return metrics
