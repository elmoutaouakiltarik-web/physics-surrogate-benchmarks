#!/usr/bin/env python3
"""Regenerate every publication figure in ``docs/img`` from the committed artifacts.

No training happens here: the plots read ``weights/*.pt``, ``benchmarks/*.json``
and the published Ghia tables, so a fresh clone reproduces the README images.

    python scripts/make_figures.py

The MMS verification figure is not produced here; it belongs to the run that
generates those metrics (``python scripts/run_benchmark_real.py --mms-only``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physics_surrogates.datasets.ghia_cavity import (  # noqa: E402
    AVAILABLE_REYNOLDS,
    get_ghia_reference,
)
from physics_surrogates.models.fno import (  # noqa: E402
    heat_solution,
    load_fno_weights,
    predict,
    random_periodic_fields,
)
from physics_surrogates.models.pinns import BurgersPINN  # noqa: E402
from physics_surrogates.reference import cole_hopf_burgers  # noqa: E402

IMG = ROOT / "docs" / "img"
WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "benchmarks"

NU_BURGERS = 0.01 / np.pi
NU_HEAT, T_HEAT = 0.01, 1.0
TEST_SAMPLES, TEST_SEED = 200, 1234
BLUE, RED, GREEN = "#1f4e9c", "#c0392b", "#1e8449"
GREY = "#5d6d7e"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "font.size": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
        }
    )


def fig_burgers() -> Path:
    """PINN vs Cole-Hopf at three times, plus the pointwise error."""
    pinn = BurgersPINN(nu=NU_BURGERS, device="cpu")
    pinn.load_weights(WEIGHTS / "burgers_pinn.pt")

    xs = np.linspace(-1.0, 1.0, 300)
    times = (0.25, 0.5, 0.75)
    colours = (BLUE, RED, GREEN)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for t, colour in zip(times, colours, strict=True):
        exact = cole_hopf_burgers(xs, t, NU_BURGERS)
        pred = pinn.predict(xs, t)
        axes[0].plot(xs, exact, color=colour, lw=2.2, label=f"$t={t}$")
        axes[0].plot(xs, pred, color=colour, lw=1.2, ls="--")
        axes[1].plot(xs, np.abs(pred - exact), color=colour, lw=1.5, label=f"$t={t}$")

    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$u(x,t)$")
    axes[0].set_title("(a) PINN (dashed) against Cole-Hopf exact (solid)")
    # Time is encoded by colour; the two line styles are named in the title, so a
    # six-entry legend is unnecessary and would cover the curves.
    axes[0].legend(title="time", loc="upper right")

    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$|u_{PINN} - u_{exact}|$")
    axes[1].set_title("(b) pointwise error: localised on the steep front")
    axes[1].set_ylim(bottom=0)

    fig.suptitle("Burgers 1D PINN - 12,737 parameters, CPU only", fontsize=11.5, fontweight="bold")
    path = IMG / "burgers_pinn_solution.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_fno_prediction() -> Path:
    """FNO prediction against the exact heat-operator solution."""
    model = load_fno_weights(WEIGHTS / "fno_heat.pt")

    u0 = random_periodic_fields(TEST_SAMPLES, 64, seed=TEST_SEED)
    exact = heat_solution(u0, NU_HEAT, T_HEAT)
    pred = predict(model, u0)
    x = np.arange(64) / 64
    sample = 0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)

    axes[0].plot(x, u0[sample], color=GREY, lw=1.1, label="initial condition $u(\\cdot,0)$")
    axes[0].plot(x, exact[sample], color=BLUE, lw=2.2, label="exact $u(\\cdot,T)$")
    axes[0].plot(x, pred[sample], color=RED, lw=1.3, ls="--", label="FNO prediction")
    axes[0].set_xlabel("$x$")
    axes[0].set_title("(a) operator applied to one input")
    axes[0].legend(loc="lower left", fontsize=7.6)

    err = np.abs(pred - exact)
    # Sorted by mean error so the panel shows structure instead of noise order.
    order = np.argsort(err.mean(axis=1))[:40]
    im = axes[1].imshow(err[order], aspect="auto", cmap="magma", origin="lower")
    axes[1].set_xlabel("grid point")
    axes[1].set_ylabel("test sample (sorted by error)")
    axes[1].set_title("(b) absolute error, 40 easiest held-out inputs")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], shrink=0.9)

    u0_fine = random_periodic_fields(TEST_SAMPLES, 256, seed=TEST_SEED)
    pred_fine = predict(model, u0_fine)
    exact_fine = heat_solution(u0_fine, NU_HEAT, T_HEAT)
    per_sample = [
        np.linalg.norm(pred[i] - exact[i]) / np.linalg.norm(exact[i]) for i in range(TEST_SAMPLES)
    ]
    per_sample_fine = [
        np.linalg.norm(pred_fine[i] - exact_fine[i]) / np.linalg.norm(exact_fine[i])
        for i in range(TEST_SAMPLES)
    ]
    bins = np.logspace(-4.6, -2.6, 26)
    axes[2].hist(per_sample, bins=bins, color=BLUE, alpha=0.75, label="64 points (train grid)")
    axes[2].hist(per_sample_fine, bins=bins, color=RED, alpha=0.55, label="256 points (zero-shot)")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("relative L2 error per sample")
    axes[2].set_ylabel("count")
    axes[2].set_title("(c) error distribution at two resolutions")
    axes[2].legend(loc="upper left")

    fig.suptitle(
        "FNO1d on the periodic heat operator - 74,209 parameters, CPU only",
        fontsize=11.5,
        fontweight="bold",
    )
    path = IMG / "fno_heat_prediction.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_fno_resolution() -> Path:
    """Relative error at three test resolutions, read from the committed JSON."""
    data = json.loads((RESULTS / "fno_heat_results.json").read_text())
    grids = list(data["test_rel_l2_by_grid"].keys())
    values = [data["test_rel_l2_by_grid"][g] for g in grids]

    fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
    bars = ax.bar([f"{g} points" for g in grids], values, color=(BLUE, "#3f6fb5", "#6f96cf"), width=0.6)
    for bar, val in zip(bars, values, strict=True):
        ax.annotate(
            f"{val:.2e}",
            (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.35)
    ax.set_ylabel("relative L2 error (200 unseen inputs)")
    ax.set_title("Same weights, three grids: zero-shot resolution transfer")
    ax.text(
        0.5,
        0.06,
        "trained on 64-point grids only",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color=GREY,
    )
    path = IMG / "fno_resolution_transfer.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_ghia() -> Path:
    """Published cavity centreline profiles (reference data only, no model)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    colours = (BLUE, RED, GREEN)

    for re, colour in zip(AVAILABLE_REYNOLDS, colours, strict=True):
        ref = get_ghia_reference(re)
        axes[0].plot(ref.u_center, ref.y_probe, "o-", color=colour, ms=4, lw=1.5, label=f"Re = {int(re)}")
        axes[1].plot(ref.v_center, ref.x_probe, "o-", color=colour, ms=4, lw=1.5, label=f"Re = {int(re)}")

    axes[0].axvline(0, color="k", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("$u / U_{lid}$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title("(a) vertical centreline $x = 0.5$ (Table I)")
    axes[0].legend(loc="lower right")

    axes[1].axvline(0, color="k", lw=0.8, alpha=0.5)
    axes[1].set_xlabel("$v / U_{lid}$")
    axes[1].set_ylabel("$x$")
    axes[1].set_title("(b) horizontal centreline $y = 0.5$ (Table II)")
    axes[1].legend(loc="lower left")

    fig.suptitle(
        "Ghia, Ghia & Shin (1982) lid-driven cavity reference data",
        fontsize=11.5,
        fontweight="bold",
    )
    path = IMG / "ghia_profiles.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    _style()

    missing = [name for name in ("burgers_pinn.pt", "fno_heat.pt") if not (WEIGHTS / name).exists()]
    if missing:
        raise SystemExit(
            f"missing weights: {', '.join(missing)}. Run the benchmarks first:\n"
            "  python scripts/run_benchmark_real.py\n"
            "  python scripts/benchmark_fno_heat.py"
        )

    written = [fig_burgers(), fig_fno_prediction(), fig_fno_resolution(), fig_ghia()]
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
