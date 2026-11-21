#!/usr/bin/env python3
"""FNO1d on the periodic heat equation: learn u(., 0) -> u(., T) against the exact FFT solution.

Writes ``benchmarks/fno_heat_results.json`` and ``weights/fno_heat.pt``. CPU, a few minutes.

    python scripts/benchmark_fno_heat.py                # published configuration
    python scripts/benchmark_fno_heat.py --epochs 20    # quick smoke run
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from physics_surrogates.models.fno import (  # noqa: E402
    evaluate,
    random_periodic_fields,
    save_fno_weights,
    train_fno_heat,
)

NU, T, N_TRAIN, N_GRID, EPOCHS, SEED = 0.01, 1.0, 1000, 64, 300, 0
TEST_SAMPLES, TEST_SEED = 200, 1234
TEST_GRIDS = (64, 128, 256)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-samples", type=int, default=N_TRAIN)
    parser.add_argument("--grid", type=int, default=N_GRID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--nu", type=float, default=NU)
    parser.add_argument("--time", type=float, default=T, dest="t_final")
    parser.add_argument(
        "--no-weights", action="store_true", help="do not write weights/fno_heat.pt"
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    model, history = train_fno_heat(
        n_train=args.train_samples,
        n_grid=args.grid,
        nu=args.nu,
        t=args.t_final,
        epochs=args.epochs,
        seed=args.seed,
    )
    train_s = time.perf_counter() - t0

    test = {
        n: evaluate(model, random_periodic_fields(TEST_SAMPLES, n, seed=TEST_SEED), args.nu, args.t_final)
        for n in TEST_GRIDS
    }

    res = {
        "problem": "periodic heat equation u_t = nu u_xx on [0,1), operator u(.,0) -> u(.,T)",
        "nu": args.nu,
        "T": args.t_final,
        "train_samples": args.train_samples,
        "train_grid": args.grid,
        "epochs": args.epochs,
        "seed": args.seed,
        "params": sum(p.numel() for p in model.parameters()),
        "train_time_s": round(train_s, 1),
        "final_train_rel_l2": history[-1],
        "test_rel_l2_by_grid": {str(k): v for k, v in test.items()},
        "reference": "exact Fourier-space solution (no solver error)",
        "test_set": f"{TEST_SAMPLES} unseen random initial conditions (seed {TEST_SEED})",
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    }

    out = ROOT / "benchmarks" / "fno_heat_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")

    if not args.no_weights:
        weights = ROOT / "weights" / "fno_heat.pt"
        weights.parent.mkdir(exist_ok=True)
        save_fno_weights(model, weights)
        print(f"weights written to {weights.relative_to(ROOT)}")

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
