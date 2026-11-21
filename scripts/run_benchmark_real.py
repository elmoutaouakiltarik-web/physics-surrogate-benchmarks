#!/usr/bin/env python3
"""Burgers 1D PINN benchmark with an exact Cole-Hopf reference.

Trains the PINN, scores it against the closed-form solution at several times,
saves the weights and writes ``benchmarks/burgers_1d_results.json``.

    python scripts/run_benchmark_real.py                 # published configuration (~27 min CPU)
    python scripts/run_benchmark_real.py --quick         # smoke run, does NOT converge
    python scripts/run_benchmark_real.py --from-weights  # re-score the committed weights
    python scripts/run_benchmark_real.py --mms-only      # metric-pipeline check only

Figures are not produced here: ``scripts/make_figures.py`` renders every image in
``docs/img`` from the committed artifacts.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physics_surrogates.datasets.ghia_cavity import (  # noqa: E402
    cavity_mms_manufactured_solution,
    evaluate_cavity_solution,
    generate_cavity_mesh,
)
from physics_surrogates.models.pinns import BurgersPINN  # noqa: E402
from physics_surrogates.reference import cole_hopf_burgers  # noqa: E402

RESULTS_DIR = ROOT / "benchmarks"
IMAGES_DIR = ROOT / "docs" / "img"
WEIGHTS_DIR = ROOT / "weights"

NU = 0.01 / np.pi
N_TEST = 200
X_TEST = np.linspace(-1.0, 1.0, N_TEST)
EVAL_TIMES = (0.25, 0.5, 0.75, 1.0)
PRIMARY_TIME = 0.5


def system_info() -> dict[str, str]:
    """Hardware/version block recorded in every result file."""
    import torch

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def score(pinn: BurgersPINN, t: float) -> dict[str, float]:
    """Relative L2, max absolute error and R^2 at time ``t`` on the centreline."""
    pred = pinn.predict(X_TEST, t)
    ref = cole_hopf_burgers(X_TEST, t, pinn.nu)
    ss_res = float(np.sum((pred - ref) ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    return {
        "l2_relative_error": float(np.linalg.norm(pred - ref) / np.linalg.norm(ref)),
        "max_absolute_error": float(np.max(np.abs(pred - ref))),
        "r2_score": float(1.0 - ss_res / (ss_tot + 1e-12)),
    }


def validation_block(pinn: BurgersPINN) -> dict:
    """Validation section of the result file, including the time profile."""
    by_time = {f"{t:.2f}": score(pinn, t) for t in EVAL_TIMES}
    primary = by_time[f"{PRIMARY_TIME:.2f}"]
    return {
        "reference": "Cole-Hopf exact solution (closed form, no solver error)",
        "n_test_points": N_TEST,
        "t": PRIMARY_TIME,
        **primary,
        "by_time": by_time,
        "note": "the metric above is at t=0.5 only; by_time shows how it degrades towards t=1",
    }


def benchmark_burgers_real(
    n_colloc: int | None = None,
    epochs_adam: int | None = None,
    epochs_lbfgs: int | None = None,
    use_rar: bool = True,
) -> dict:
    """Train the PINN, score it, persist weights and JSON."""
    print("=" * 72)
    print("BURGERS 1D PINN - exact Cole-Hopf validation")
    print("=" * 72)

    info = system_info()
    print(f"system: {info['platform']} | {info['device']} | torch {info['torch']}")

    n_colloc = n_colloc or 10000
    epochs_adam = epochs_adam or 10000
    epochs_lbfgs = epochs_lbfgs or 3000
    n_ic, n_bc = 256, 256

    pinn = BurgersPINN(nu=NU, device=info["device"], activation="tanh")
    print(
        f"training: Adam {epochs_adam} ep + L-BFGS {epochs_lbfgs} it, "
        f"{n_colloc} collocation points, RAR={use_rar}"
    )
    start = time.time()
    metrics = pinn.train(
        n_colloc=n_colloc,
        n_ic=n_ic,
        n_bc=n_bc,
        epochs_adam=epochs_adam,
        epochs_lbfgs=epochs_lbfgs,
        lr=1e-3,
        use_rar=use_rar,
        rar_interval=2000,
    )
    elapsed = time.time() - start
    print(f"training finished in {elapsed:.1f}s")

    validation = validation_block(pinn)
    print(f"t={PRIMARY_TIME}: rel L2 {validation['l2_relative_error']:.4e}, "
          f"R2 {validation['r2_score']:.6f}")

    WEIGHTS_DIR.mkdir(exist_ok=True)
    weight_path = WEIGHTS_DIR / "burgers_pinn.pt"
    pinn.save_weights(weight_path)

    results = {
        "case": "Burgers1D",
        "pde": "u_t + u*u_x = nu*u_xx",
        "nu": float(NU),
        "domain": {"x": [-1, 1], "t": [0, 1]},
        "initial_condition": "u(x,0) = -sin(pi*x)",
        "boundary_conditions": "u(-1,t) = u(1,t) = 0",
        "network": {"layers": [2, 64, 64, 64, 64, 1], "activation": "tanh"},
        "training": {
            "n_colloc": int(n_colloc),
            "n_colloc_final": int(metrics["n_colloc_final"]),
            "n_ic": int(n_ic),
            "n_bc": int(n_bc),
            "epochs_adam": int(epochs_adam),
            "epochs_lbfgs": int(epochs_lbfgs),
            "use_rar": bool(use_rar),
            "rar_interval": 2000 if use_rar else None,
            "rar_points_per_interval": 1000 if use_rar else None,
        },
        "model": {"class": "BurgersPINN", "params": int(metrics["n_params"])},
        "system": info,
        "metrics": {
            "final_loss": float(metrics["final_loss"]),
            "elapsed_time_sec": float(elapsed),
        },
        "validation": validation,
        "weights_file": weight_path.relative_to(ROOT).as_posix(),
        "weights_size_mb": round(weight_path.stat().st_size / (1024 * 1024), 4),
        "figures": ["docs/img/burgers_pinn_solution.png"],
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "burgers_1d_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results written to {out.relative_to(ROOT)}")
    return results


def refresh_from_weights() -> dict:
    """Re-score the committed weights and refresh the validation block.

    The trained weights are the published artifact, so re-scoring them is the
    honest way to update the reported metrics on another machine without a
    27-minute retrain. Training metadata is preserved from the recorded run.
    """
    out = RESULTS_DIR / "burgers_1d_results.json"
    results = json.loads(out.read_text())

    pinn = BurgersPINN(nu=float(results["nu"]), device="cpu")
    pinn.load_weights(ROOT / results["weights_file"])

    results["model"] = {"class": "BurgersPINN", "params": sum(p.numel() for p in pinn.net.parameters())}
    results["validation"] = validation_block(pinn)
    results["weights_size_mb"] = round((ROOT / results["weights_file"]).stat().st_size / 1e6, 4)
    results["rescored_from_weights"] = True
    out.write_text(json.dumps(results, indent=2) + "\n")

    v = results["validation"]
    print(f"re-scored committed weights: rel L2 {v['l2_relative_error']:.6e}, "
          f"R2 {v['r2_score']:.6f}")
    print(f"written to {out.relative_to(ROOT)}")
    return results


def benchmark_mms_cavity() -> dict:
    """MMS metric-pipeline check (verification, not a model result)."""
    print("=" * 72)
    print("MMS CAVITY - metric pipeline check (no model involved)")
    print("=" * 72)

    nx = ny = 65
    x, y, X, Y = generate_cavity_mesh(nx=nx, ny=ny)
    u_mms, v_mms, _ = cavity_mms_manufactured_solution(X, Y)

    rng = np.random.default_rng(42)
    sigma = 0.01
    u_pred = u_mms + sigma * rng.standard_normal(u_mms.shape)
    v_pred = v_mms + sigma * rng.standard_normal(v_mms.shape)

    metrics = evaluate_cavity_solution(u_pred, v_pred, u_mms, v_mms)
    print(f"synthetic noise sigma={sigma}: L2(u) {metrics['l2_u_relative']:.4e}, "
          f"R2(u) {metrics['r2_u']:.6f}")

    _plot_mms(X, Y, u_mms, v_mms, u_pred, v_pred)

    results = {
        "case": "MMS_Cavity",
        "method": "Method of Manufactured Solutions (verification only)",
        "mesh": {"nx": nx, "ny": ny},
        "perturbation": {"type": "gaussian noise", "sigma": sigma, "seed": 42},
        "field": (
            "u = d(psi)/dy, v = -d(psi)/dx, psi = 16 x^2 (1-x)^2 y^2 (1-y)^2 "
            "(analytically divergence free, no-slip on all walls)"
        ),
        "note": "Checks the error pipeline on a known field. This is NOT a model result.",
        "metrics": metrics,
    }
    out = RESULTS_DIR / "mms_cavity_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"written to {out.relative_to(ROOT)}")
    return results


def _plot_mms(X, Y, u_mms, v_mms, u_pred, v_pred) -> None:
    """Render the MMS verification figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True)
    panels = (
        (u_mms, "MMS $u$", "RdBu_r"),
        (u_pred, "noisy $u$ (sigma = 0.01)", "RdBu_r"),
        (np.abs(u_pred - u_mms), "$|u_{noisy} - u_{MMS}|$", "magma"),
        (v_mms, "MMS $v$", "RdBu_r"),
        (v_pred, "noisy $v$ (sigma = 0.01)", "RdBu_r"),
        (np.abs(v_pred - v_mms), "$|v_{noisy} - v_{MMS}|$", "magma"),
    )
    for ax, (field, title, cmap) in zip(axes.ravel(), panels, strict=True):
        im = ax.contourf(X, Y, field, levels=24, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        "MMS verification: error pipeline on a known field (not a model result)",
        fontsize=12,
        fontweight="bold",
    )
    path = IMAGES_DIR / "mms_cavity_verification.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"figure written to {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--colloc", type=int, default=None, help="collocation points")
    parser.add_argument("--adam", type=int, default=None, help="Adam epochs")
    parser.add_argument("--lbfgs", type=int, default=None, help="L-BFGS iterations")
    parser.add_argument("--no-rar", action="store_true", help="disable residual adaptive refinement")
    parser.add_argument("--quick", action="store_true", help="fast smoke configuration (CI only)")
    parser.add_argument(
        "--from-weights",
        action="store_true",
        help="skip training and re-score the committed weights",
    )
    parser.add_argument("--mms-only", action="store_true", help="run only the MMS metric check")
    args = parser.parse_args()

    if args.from_weights:
        refresh_from_weights()
        return
    if args.mms_only:
        benchmark_mms_cavity()
        return

    if args.quick:
        args.colloc = args.colloc or 2000
        args.adam = args.adam or 400
        args.lbfgs = args.lbfgs or 40

    benchmark_burgers_real(
        n_colloc=args.colloc,
        epochs_adam=args.adam,
        epochs_lbfgs=args.lbfgs,
        use_rar=not args.no_rar,
    )
    benchmark_mms_cavity()


if __name__ == "__main__":
    main()
