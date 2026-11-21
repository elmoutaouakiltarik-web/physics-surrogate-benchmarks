"""Lid-driven cavity benchmark data and helpers.

Reference
---------
Ghia, U., Ghia, K. N., & Shin, C. T. (1982). "High-Re Solutions for Incompressible
Flow Using the Navier-Stokes Equations and a Multigrid Method." Journal of
Computational Physics, 48(3), 387-411.

The arrays below are the published tables:

* Table I  - ``u`` along the vertical centreline ``x = 0.5``, as a function of ``y``.
* Table II - ``v`` along the horizontal centreline ``y = 0.5``, as a function of ``x``.

Reynolds numbers 100, 400 and 1000 are transcribed. Values are normalised by the
lid velocity and the cavity side, so ``u`` is O(1) and ``v`` is O(0.1-0.5).

Data caveats, recorded rather than silently "fixed":

* Re = 400, Table II at x = 0.9063 is printed as -0.23827 in the paper; it breaks
  the monotone trend and is widely considered a typesetting error. The printed
  value is kept and flagged in ``GHIA_KNOWN_MISPRINTS``.
* Re = 400, Table II at x = 0.5 is printed 0.05188 in some scans and 0.05186 in
  the dataset distributed with PetIBM. The difference (2e-5) is below the
  precision of the paper's own grid; 0.05186 is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Reynolds numbers with transcribed published data.
AVAILABLE_REYNOLDS: tuple[float, ...] = (100.0, 400.0, 1000.0)

#: Table I y-grid (vertical centreline, x = 0.5).
_GHIA_Y = np.array(
    [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
        0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
    ]
)

#: Table II x-grid (horizontal centreline, y = 0.5).
_GHIA_X = np.array(
    [
        1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
        0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000,
    ]
)

_GHIA_U: dict[int, np.ndarray] = {
    100: np.array(
        [
            1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
            -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
            -0.04775, -0.04192, -0.03717, 0.00000,
        ]
    ),
    400: np.array(
        [
            1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256,
            0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612,
            -0.10338, -0.09266, -0.08186, 0.00000,
        ]
    ),
    1000: np.array(
        [
            1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719,
            0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730,
            -0.22220, -0.20196, -0.18109, 0.00000,
        ]
    ),
}

_GHIA_V: dict[int, np.ndarray] = {
    100: np.array(
        [
            0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
            -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890,
            0.10091, 0.09233, 0.00000,
        ]
    ),
    400: np.array(
        [
            0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993,
            -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920,
            0.19713, 0.18360, 0.00000,
        ]
    ),
    1000: np.array(
        [
            0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665,
            -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353,
            0.29012, 0.27485, 0.00000,
        ]
    ),
}

#: Values kept as printed in the paper but known to be suspect.
GHIA_KNOWN_MISPRINTS: dict[str, str] = {
    "Re=400/TableII/x=0.9063": "-0.23827 is printed in the paper but breaks the trend; kept verbatim",
}


@dataclass
class GhiaReference:
    """Ghia et al. (1982) reference profiles for one Reynolds number."""

    re: float
    y_probe: np.ndarray  # y on the vertical centreline x = 0.5 (Table I)
    u_center: np.ndarray  # u(y) / U_lid along x = 0.5 (Table I)
    x_probe: np.ndarray  # x on the horizontal centreline y = 0.5 (Table II)
    v_center: np.ndarray  # v(x) / U_lid along y = 0.5 (Table II)


def available_reynolds() -> tuple[float, ...]:
    """Reynolds numbers for which published reference data is transcribed."""
    return AVAILABLE_REYNOLDS


def get_ghia_reference(re: float) -> GhiaReference:
    """Return the published profiles for ``re`` (100, 400 or 1000)."""
    key = int(re)
    if key not in _GHIA_U or key != re:
        raise ValueError(
            f"Ghia reference data available for Re in {list(AVAILABLE_REYNOLDS)}, got Re={re!r}"
        )
    return GhiaReference(
        re=float(re),
        y_probe=_GHIA_Y.copy(),
        u_center=_GHIA_U[key].copy(),
        x_probe=_GHIA_X.copy(),
        v_center=_GHIA_V[key].copy(),
    )


def generate_cavity_mesh(
    nx: int = 129, ny: int = 129
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform Cartesian grid on the unit cavity.

    Returns:
        x, y: coordinate vectors of length nx, ny.
        X, Y: meshgrids of shape [nx, ny] (``indexing="ij"``).
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y


def cavity_bc_mask(
    nx: int = 129, ny: int = 129
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks for the four cavity walls.

    Boundary conditions: the lid (y = 1) moves with u = 1, v = 0; the other three
    walls are no-slip (u = v = 0).
    """
    top = np.zeros((nx, ny), dtype=bool)
    top[-1, :] = True
    bottom = np.zeros((nx, ny), dtype=bool)
    bottom[0, :] = True
    left = np.zeros((nx, ny), dtype=bool)
    left[:, 0] = True
    right = np.zeros((nx, ny), dtype=bool)
    right[:, -1] = True
    return top, bottom, left, right


def cavity_mms_streamfunction(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Stream function psi = 16 x^2 (1-x)^2 y^2 (1-y)^2 of the manufactured field."""
    return 16.0 * X**2 * (1.0 - X) ** 2 * Y**2 * (1.0 - Y) ** 2


def cavity_mms_manufactured_solution(
    X: np.ndarray, Y: np.ndarray, t: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Manufactured velocity field for Method of Manufactured Solutions (MMS).

    The velocity is taken as ``u = d(psi)/dy``, ``v = -d(psi)/dx`` with
    ``psi = 16 x^2 (1-x)^2 y^2 (1-y)^2``, so the discrete field is *analytically
    divergence free* and no-slip on all four walls. With ``A(s) = s^2 (1-s)^2``
    and ``B(s) = A'(s) = 2 s (1-s) (1-2s)``:

        u = 16 A(x) B(y),   v = -16 A(y) B(x),   du/dx + dv/dy = 0 exactly.

    The pressure is an arbitrary smooth field: MMS verification checks that a
    solver reproduces a known solution, not that the field is a physical flow.

    Args:
        X, Y: coordinate grids on the unit square.
        t: unused (the manufactured field is steady); kept for API symmetry.

    Returns:
        ``(u, v, p)`` arrays with the shape of ``X``.
    """
    u = 16.0 * X**2 * (1.0 - X) ** 2 * 2.0 * Y * (1.0 - Y) * (1.0 - 2.0 * Y)
    v = -16.0 * Y**2 * (1.0 - Y) ** 2 * 2.0 * X * (1.0 - X) * (1.0 - 2.0 * X)
    p = 10.0 * np.sin(np.pi * X) * np.cos(np.pi * Y)
    return u, v, p


def evaluate_cavity_solution(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray | None = None,
    v_ref: np.ndarray | None = None,
) -> dict:
    """Error metrics for a predicted velocity field.

    Metrics (per component and averaged):
      * relative L2 ``||pred - ref||_2 / ||ref||_2``
      * R^2 ``1 - SS_res / SS_tot``
      * maximum absolute error

    Returns an empty dict when no reference is supplied.
    """
    metrics: dict[str, float] = {}
    if u_ref is None or v_ref is None:
        return metrics

    u_ref_f = np.asarray(u_ref, dtype=float).ravel()
    v_ref_f = np.asarray(v_ref, dtype=float).ravel()
    u_pred_f = np.asarray(u_pred, dtype=float).ravel()
    v_pred_f = np.asarray(v_pred, dtype=float).ravel()

    if not (u_pred_f.shape == u_ref_f.shape and v_pred_f.shape == v_ref_f.shape):
        raise ValueError(
            "predicted and reference fields must have the same shape, got "
            f"u {u_pred_f.shape} vs {u_ref_f.shape}, v {v_pred_f.shape} vs {v_ref_f.shape}"
        )

    l2_u = np.linalg.norm(u_pred_f - u_ref_f) / (np.linalg.norm(u_ref_f) + 1e-12)
    l2_v = np.linalg.norm(v_pred_f - v_ref_f) / (np.linalg.norm(v_ref_f) + 1e-12)

    def r2(pred: np.ndarray, ref: np.ndarray) -> float:
        ss_res = np.sum((ref - pred) ** 2)
        ss_tot = np.sum((ref - np.mean(ref)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))

    metrics.update(
        {
            "l2_u_relative": float(l2_u),
            "l2_v_relative": float(l2_v),
            "l2_relative_mean": float((l2_u + l2_v) / 2),
            "r2_u": r2(u_pred_f, u_ref_f),
            "r2_v": r2(v_pred_f, v_ref_f),
            "r2_mean": (r2(u_pred_f, u_ref_f) + r2(v_pred_f, v_ref_f)) / 2,
            "max_error_u": float(np.max(np.abs(u_pred_f - u_ref_f))),
            "max_error_v": float(np.max(np.abs(v_pred_f - v_ref_f))),
        }
    )
    return metrics


def ghia_centerline_errors(
    u_field: np.ndarray,
    v_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    re: float,
) -> dict[str, float]:
    """Compare a cavity solution on a Cartesian grid with the Ghia profiles.

    ``u_field`` and ``v_field`` are indexed ``[i, j]`` with ``i`` along ``x``
    (matching :func:`generate_cavity_mesh`). The centreline values are linearly
    interpolated onto the published probe locations.

    Returns relative L2 errors (in percent) for the two centreline profiles.
    """
    u_field = np.asarray(u_field, dtype=float)
    v_field = np.asarray(v_field, dtype=float)
    if u_field.shape != (len(x), len(y)) or v_field.shape != (len(x), len(y)):
        raise ValueError(
            f"fields must have shape {(len(x), len(y))}, got {u_field.shape} and {v_field.shape}"
        )

    ref = get_ghia_reference(re)

    j_mid = int(np.argmin(np.abs(x - 0.5)))
    u_line = np.interp(ref.y_probe, y, u_field[j_mid, :])
    u_err = 100.0 * np.linalg.norm(u_line - ref.u_center) / np.linalg.norm(ref.u_center)

    i_mid = int(np.argmin(np.abs(y - 0.5)))
    v_line = np.interp(ref.x_probe, x, v_field[:, i_mid])
    v_err = 100.0 * np.linalg.norm(v_line - ref.v_center) / np.linalg.norm(ref.v_center)

    return {"u_centerline_l2_percent": float(u_err), "v_centerline_l2_percent": float(v_err)}
