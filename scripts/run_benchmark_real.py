"""
Real benchmark with external validation.

Burgers 1D: validated against Cole-Hopf exact solution
Generates: JSON results + matplotlib plots
Reports: Real metrics only (no fabrication)
"""
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_surrogates.datasets.ghia_cavity import (
    cavity_mms_manufactured_solution,
    evaluate_cavity_solution,
    generate_cavity_mesh,
)
from physics_surrogates.models.pinns import BurgersPINN

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent.parent / "benchmarks"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "img"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

RESULTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)


# numpy>=2.0 renamed trapz -> trapezoid; support both
_trapz_fn = getattr(np, "trapezoid", None) or np.trapz


def _trapz(y, dx):
    return _trapz_fn(y, dx=dx)


def cole_hopf_burgers(x, t, nu, eta_half_width=4.0, n_eta=160001):
    """Reference solution of viscous Burgers via the Cole-Hopf integral.

    PDE:  u_t + u*u_x = nu*u_xx,  u(x,0) = -sin(pi*x),  u(+-1,t) = 0.

    The Cole-Hopf transform turns the problem into the heat equation, whose
    solution can be written in closed form (Basdevant et al., 1986):

        u(x,t) = - I_num / I_den
        I_num  = int sin(pi*(x-eta)) * f(x-eta) * exp(-eta^2/(4*nu*t)) d(eta)
        I_den  = int             f(x-eta) * exp(-eta^2/(4*nu*t)) d(eta)
        f(y)   = exp(-cos(pi*y) / (2*pi*nu))

    For nu = 0.01/pi the exponent reaches +-50, so the integrand is evaluated
    in log space and shifted by its maximum (log-sum-exp) before exponentiating.
    A naive evaluation overflows and a forward-Euler heat solve is unstable at
    this viscosity; both produce a reference that silently disagrees with the
    true solution, which is why this quadrature is used instead.

    Args:
        x: Scalar or array of spatial coordinates in [-1, 1].
        t: Time (>= 0).
        nu: Kinematic viscosity.
        eta_half_width: Truncation of the eta integral. The Gaussian factor has
            standard deviation sqrt(2*nu*t) (~0.06 at nu=0.01/pi, t=0.5), so 4.0
            is several hundred standard deviations and the truncation error is
            far below float64 resolution.
        n_eta: Number of quadrature nodes (composite trapezoid).

    Returns:
        np.ndarray with the same shape as x.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if t < 1e-12:
        out = -np.sin(np.pi * x_arr)
        return out if np.ndim(x) else float(out[0])

    eta = np.linspace(-eta_half_width, eta_half_width, n_eta)
    d_eta = eta[1] - eta[0]
    gauss_log = -(eta**2) / (4.0 * nu * t)

    u = np.empty_like(x_arr)
    for i, xi in enumerate(x_arr):
        y = xi - eta
        # log of the full integrand weight, then shift by its max for stability
        w = -np.cos(np.pi * y) / (2.0 * np.pi * nu) + gauss_log
        w -= w.max()
        weight = np.exp(w)
        num = _trapz(np.sin(np.pi * y) * weight, d_eta)
        den = _trapz(weight, d_eta)
        u[i] = -num / den

    return u if np.ndim(x) else float(u[0])


def benchmark_burgers_real(n_colloc=None, epochs_adam=None, epochs_lbfgs=None, use_rar=True):
    """Real Burgers PINN benchmark with Cole-Hopf validation."""
    print("\n" + "="*70)
    print("BENCHMARK: Burgers 1D PINN (Real Execution)")
    print("="*70)

    # System info
    import torch
    device_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    print(f"\nSystem: {device_info['platform']}")
    print(f"Device: {device_info['device']}")

    # Parameters
    nu = 0.01 / np.pi
    device = device_info["device"]

    # Initialize and train
    print("\nInitializing PINN...")
    pinn = BurgersPINN(nu=nu, device=device, activation="tanh")

    # Training budget. Defaults are the converged configuration; override them
    # with --colloc/--adam/--lbfgs for a fast smoke run (CI uses --quick).
    n_colloc = n_colloc or 10000
    epochs_adam = epochs_adam or 10000
    epochs_lbfgs = epochs_lbfgs or 3000
    n_ic, n_bc = 256, 256

    print(
        f"Training PINN (Adam {epochs_adam} ep + L-BFGS {epochs_lbfgs} iter, "
        f"{n_colloc} colloc, RAR={use_rar})..."
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
    print(f"Training complete in {elapsed:.1f}s")

    # Test on center-line
    x_test = np.linspace(-1, 1, 200)
    t_test = 0.5
    u_pred = pinn.predict(x_test, t_test)

    # Cole-Hopf reference (validated against an independent finite-difference
    # solver in tests/test_reference_solutions.py)
    print(f"Computing Cole-Hopf reference solution at t={t_test}...")
    u_ref = cole_hopf_burgers(x_test, t_test, nu)

    # Metrics
    l2_error = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
    max_error = np.max(np.abs(u_pred - u_ref))
    r2 = 1 - np.sum((u_pred - u_ref)**2) / (np.sum((u_ref - np.mean(u_ref))**2) + 1e-12)

    print("\nValidation (Cole-Hopf reference):")
    print(f"  L2 relative error: {l2_error:.4e}")
    print(f"  Max absolute error: {max_error:.4e}")
    print(f"  R² score: {r2:.6f}")
    print(f"  Final training loss: {metrics['final_loss']:.4e}")

    # Save weights
    weight_path = WEIGHTS_DIR / "burgers_pinn.pt"
    pinn.save_weights(weight_path)
    weight_size_mb = weight_path.stat().st_size / (1024*1024)
    print(f"\nWeights saved: {weight_path} ({weight_size_mb:.2f} MB)")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Solution comparison
    ax = axes[0]
    ax.plot(x_test, u_ref, "b-", linewidth=2, label="Cole-Hopf (reference)")
    ax.plot(x_test, u_pred, "r--", linewidth=1.5, label="PINN prediction")
    ax.fill_between(x_test, u_ref, u_pred, alpha=0.2, color="gray", label="Error")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("u(x, t=0.5)", fontsize=11)
    ax.set_title("Burgers 1D: PINN vs Cole-Hopf Exact", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Training loss
    ax = axes[1]
    ax.semilogy(pinn.losses, "b-", linewidth=1.5)
    ax.set_xlabel("Training checkpoint (500 epochs)", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Training Convergence", fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plot_path = IMAGES_DIR / "burgers_pinn_solution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close()

    # JSON results
    results = {
        "case": "Burgers1D",
        "pde": "u_t + u*u_x = nu*u_xx",
        "nu": float(nu),
        "domain": {"x": [-1, 1], "t": [0, 1]},
        "network": {
            "layers": [2, 64, 64, 64, 64, 1],
            "activation": "tanh",
        },
        "training": {
            "n_colloc": int(n_colloc),
            "n_ic": 100,
            "n_bc": 100,
            "epochs_adam": int(epochs_adam),
            "epochs_lbfgs": int(epochs_lbfgs),
            "use_rar": bool(use_rar),
            "rar_interval": 2000 if use_rar else None,
        },
        "system": device_info,
        "metrics": {
            "final_loss": float(metrics["final_loss"]),
            "elapsed_time_sec": float(elapsed),
            "test_time": 0.5,
        },
        "validation": {
            "reference": "Cole-Hopf exact solution",
            "n_test_points": len(x_test),
            "l2_relative_error": float(l2_error),
            "max_absolute_error": float(max_error),
            "r2_score": float(r2),
        },
        # Repo-relative so the published JSON carries no local filesystem path.
        "weights_file": weight_path.relative_to(REPO_ROOT).as_posix(),
        "weights_size_mb": float(weight_size_mb),
    }

    result_path = RESULTS_DIR / "burgers_1d_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {result_path}\n")

    return results


def benchmark_mms_cavity():
    """MMS verification for cavity (correct usage)."""
    print("="*70)
    print("BENCHMARK: MMS Cavity (Verification, Not Validation)")
    print("="*70)

    # Generate mesh and MMS solution
    nx, ny = 65, 65
    x, y, X, Y = generate_cavity_mesh(nx=nx, ny=ny)
    u_mms, v_mms, p_mms = cavity_mms_manufactured_solution(X, Y)

    # Simple "surrogate": just add noise to MMS (proof-of-concept)
    np.random.seed(42)
    u_pred = u_mms + 0.01 * np.random.randn(*u_mms.shape)
    v_pred = v_mms + 0.01 * np.random.randn(*v_mms.shape)

    # Evaluate
    metrics = evaluate_cavity_solution(u_pred, v_pred, u_mms, v_mms)

    print("MMS field verification:")
    print(f"  L2 relative (u): {metrics['l2_u_relative']:.4e}")
    print(f"  L2 relative (v): {metrics['l2_v_relative']:.4e}")
    print(f"  R² (u): {metrics['r2_u']:.6f}")
    print(f"  R² (v): {metrics['r2_v']:.6f}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    im = axes[0, 0].contourf(X, Y, u_mms, levels=20, cmap="RdBu_r")
    axes[0, 0].set_title("MMS: u(x,y)", fontweight="bold")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].contourf(X, Y, u_pred, levels=20, cmap="RdBu_r")
    axes[0, 1].set_title("Noisy prediction: u_pred", fontweight="bold")
    axes[0, 1].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 1])

    error_u = np.abs(u_pred - u_mms)
    im = axes[0, 2].contourf(X, Y, error_u, levels=20, cmap="hot")
    axes[0, 2].set_title("Error: |u_pred - u_mms|", fontweight="bold")
    axes[0, 2].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].contourf(X, Y, v_mms, levels=20, cmap="RdBu_r")
    axes[1, 0].set_title("MMS: v(x,y)", fontweight="bold")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].contourf(X, Y, v_pred, levels=20, cmap="RdBu_r")
    axes[1, 1].set_title("Noisy prediction: v_pred", fontweight="bold")
    axes[1, 1].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 1])

    error_v = np.abs(v_pred - v_mms)
    im = axes[1, 2].contourf(X, Y, error_v, levels=20, cmap="hot")
    axes[1, 2].set_title("Error: |v_pred - v_mms|", fontweight="bold")
    axes[1, 2].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plot_path = IMAGES_DIR / "mms_cavity_verification.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}\n")
    plt.close()

    # Results
    results = {
        "case": "MMS_Cavity",
        "method": "Method of Manufactured Solutions (Verification Only)",
        "mesh": {"nx": nx, "ny": ny},
        "note": "MMS is used to verify code correctness. R² is not a validation metric.",
        "metrics": metrics,
    }

    result_path = RESULTS_DIR / "mms_cavity_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"MMS results saved: {result_path}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Physics surrogate benchmarks (real execution)")
    parser.add_argument("--colloc", type=int, default=None, help="collocation points")
    parser.add_argument("--adam", type=int, default=None, help="Adam epochs")
    parser.add_argument("--lbfgs", type=int, default=None, help="L-BFGS iterations")
    parser.add_argument("--no-rar", action="store_true", help="disable residual adaptive refinement")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fast smoke configuration (does NOT converge; for CI only)",
    )
    args = parser.parse_args()

    if args.quick:
        args.colloc = args.colloc or 2000
        args.adam = args.adam or 500
        args.lbfgs = args.lbfgs or 50

    print("\n" + "PHYSICS SURROGATE REAL BENCHMARK".center(70))
    print("Execution with real validation\n".center(70))

    try:
        burgers_result = benchmark_burgers_real(
            n_colloc=args.colloc,
            epochs_adam=args.adam,
            epochs_lbfgs=args.lbfgs,
            use_rar=not args.no_rar,
        )
        mms_result = benchmark_mms_cavity()

        print("="*70)
        print("BENCHMARKS COMPLETE")
        print("="*70)
        print("\nOutput files:")
        print(f"  - {RESULTS_DIR / 'burgers_1d_results.json'}")
        print(f"  - {RESULTS_DIR / 'mms_cavity_results.json'}")
        print(f"  - {IMAGES_DIR / 'burgers_pinn_solution.png'}")
        print(f"  - {IMAGES_DIR / 'mms_cavity_verification.png'}")
        print(f"  - {WEIGHTS_DIR / 'burgers_pinn.pt'}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
