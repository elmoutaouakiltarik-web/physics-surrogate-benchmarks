#!/usr/bin/env python3
"""FNO1d vs dense MLP on nonlinear viscous Burgers: u(., 0) -> u(., T), zero-shot to finer grids.

Writes benchmarks/fno_burgers_results.json (mean and std over seeds). CPU only.
Reference = pseudo-spectral solver (numerical, validated in tests/test_solvers.py).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physics_surrogates.models.burgers_operator import (  # noqa: E402
    MLPOperator,
    build_fno,
    count_params,
    evaluate_model,
    identity_error,
    make_dataset,
    train_operator,
)
from physics_surrogates.models.fno import heat_solution, relative_l2  # noqa: E402

NU, T, AMP = 0.02, 0.5, 2.0
N_TRAIN, TRAIN_GRID, EPOCHS = 1000, 64, 200
SEEDS = (0, 1, 2)
TEST_GRIDS = (64, 128, 256)
TEST_SEED, N_TEST = 1234, 200


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def main() -> None:
    test = {n: make_dataset(N_TEST, n, NU, T, TEST_SEED, AMP) for n in TEST_GRIDS}
    fno_runs: dict[int, list[float]] = {n: [] for n in TEST_GRIDS}
    mlp_runs: list[float] = []
    times: dict[str, list[float]] = {"fno": [], "mlp": []}
    for seed in SEEDS:
        u0, ut = make_dataset(N_TRAIN, TRAIN_GRID, NU, T, seed, AMP)
        import torch

        torch.manual_seed(seed)
        fno = build_fno()
        t0 = time.perf_counter()
        train_operator(fno, u0, ut, epochs=EPOCHS, seed=seed)
        times["fno"].append(time.perf_counter() - t0)
        for n in TEST_GRIDS:
            fno_runs[n].append(evaluate_model(fno, *test[n]))
        torch.manual_seed(seed)
        mlp = MLPOperator(TRAIN_GRID, hidden=256, layers=3)
        t0 = time.perf_counter()
        train_operator(mlp, u0, ut, epochs=EPOCHS, seed=seed)
        times["mlp"].append(time.perf_counter() - t0)
        mlp_runs.append(evaluate_model(mlp, *test[TRAIN_GRID]))
        print(f"seed {seed}: fno@64={fno_runs[64][-1]:.4f} fno@256={fno_runs[256][-1]:.4f} "
              f"mlp@64={mlp_runs[-1]:.4f}", flush=True)

    res = {
        "problem": "periodic viscous Burgers u_t + u u_x = nu u_xx on [0,1), operator u(.,0) -> u(.,T)",
        "nu": NU, "T": T, "ic_amplitude": AMP,
        "train_samples": N_TRAIN, "train_grid": TRAIN_GRID, "epochs": EPOCHS, "seeds": list(SEEDS),
        "test_set": f"{N_TEST} unseen random initial conditions (seed {TEST_SEED}), reference "
                    "recomputed with the spectral solver at each test grid",
        "params": {"fno": count_params(build_fno()), "mlp": count_params(MLPOperator(TRAIN_GRID))},
        "test_rel_l2": {
            "fno_by_grid": {str(n): mean_std(v) for n, v in fno_runs.items()},
            "mlp_train_grid": mean_std(mlp_runs),
            "mlp_other_grids": "not applicable (fixed input size)",
        },
        "baselines_rel_l2": {
            str(n): {
                "identity_u0_as_prediction": identity_error(*test[n]),
                "linear_heat_theory_ignoring_advection": relative_l2(
                    heat_solution(test[n][0], NU, T), test[n][1]
                ),
            }
            for n in TEST_GRIDS
        },
        "train_time_s_per_seed": {k: round(float(np.mean(v)), 1) for k, v in times.items()},
        "reference": "pseudo-spectral IFRK4 solver (numerical); resolution check: grid-64 vs grid-256 "
                     "reference differ by ~2e-3 relative (max-norm)",
    }
    out = ROOT / "benchmarks" / "fno_burgers_results.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
