#!/usr/bin/env python3
"""Is the Burgers FNO actually cheaper than the solver it imitates? Accuracy-matched timing.

A surrogate speed-up is only meaningful against a solver run at the *same* accuracy, so the
pseudo-spectral solver is swept over time steps and scored exactly like the FNO (relative L2
against the dt = 1e-3 reference on the same 200 unseen initial conditions). For every grid
the script reports:

* the FNO error and wall time (batch of 200 and single query);
* the solver error and wall time for each dt;
* the cheapest stable dt whose error is at or below the FNO error ("accuracy-matched");
* the break-even query count once training and training-data generation are paid for.

Wall times are medians over repeats on one CPU core (``torch.set_num_threads(1)``; NumPy's
FFT is single-threaded), so the two sides use the same hardware budget. Needs the weights
written by ``scripts/benchmark_fno_burgers.py``. Writes benchmarks/speed_burgers_results.json.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physics_surrogates.models.burgers_operator import make_dataset  # noqa: E402
from physics_surrogates.models.fno import load_fno_weights, predict, relative_l2  # noqa: E402
from physics_surrogates.solvers import burgers_solve  # noqa: E402

NU, T, AMP = 0.02, 0.5, 2.0
REF_DT = 1e-3
GRIDS = (64, 128, 256)
# Dense near the explicit stability limit (~dt = 1e-2 here) so the cheapest stable step is
# located to within 1e-3; a coarse sweep would overstate the surrogate's speed-up.
DTS = (1e-3, 2e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 5e-2, 1e-1)
TEST_SEED, N_TEST = 1234, 200
N_TRAIN, TRAIN_GRID = 1000, 64
WEIGHTS = ROOT / "weights" / "fno_burgers.pt"
REPEATS = 7


def median_time(fn, repeats: int = REPEATS) -> float:
    with np.errstate(all="ignore"):  # unstable dt sweeps overflow by design
        return _median_time(fn, repeats)


def _median_time(fn, repeats: int) -> float:
    fn()  # warm-up (allocations, FFT plans, torch kernels)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def solver_error(u0: np.ndarray, ref: np.ndarray, dt: float) -> float:
    with np.errstate(all="ignore"):
        out = burgers_solve(u0, NU, T, dt=dt)
    if not np.all(np.isfinite(out)):
        return float("inf")
    return relative_l2(out, ref)


def main() -> None:
    torch.set_num_threads(1)
    model = load_fno_weights(WEIGHTS)
    train_meta = json.loads((ROOT / "benchmarks" / "fno_burgers_results.json").read_text(encoding="utf-8"))
    train_s = float(train_meta["train_time_s_per_seed"]["fno"])

    # Cost of producing the training set (1,000 reference solves on the training grid).
    data_s = median_time(lambda: make_dataset(N_TRAIN, TRAIN_GRID, NU, T, 0, AMP), repeats=3)

    by_grid: dict[str, dict] = {}
    for n in GRIDS:
        u0, ref = make_dataset(N_TEST, n, NU, T, TEST_SEED, AMP)
        fno_err = relative_l2(predict(model, u0), ref)
        fno_batch = median_time(lambda u0=u0: predict(model, u0))
        fno_single = median_time(lambda u0=u0: predict(model, u0[:1]), repeats=50)

        sweep = []
        for dt in DTS:
            err = solver_error(u0, ref, dt)
            wall = median_time(lambda u0=u0, dt=dt: burgers_solve(u0, NU, T, dt=dt), repeats=3)
            single = median_time(lambda u0=u0, dt=dt: burgers_solve(u0[:1], NU, T, dt=dt), repeats=5)
            sweep.append({"dt": dt, "rel_l2": err, "batch_s": wall, "single_s": single,
                          "stable": bool(np.isfinite(err))})
        matched = [s for s in sweep if s["rel_l2"] <= fno_err]
        best = min(matched, key=lambda s: s["batch_s"]) if matched else None

        entry = {
            "fno": {"rel_l2": fno_err, "batch_s": fno_batch, "single_s": fno_single},
            "solver_sweep": sweep,
            "accuracy_matched_solver": best,
        }
        if best is not None:
            entry["speedup_batch"] = best["batch_s"] / fno_batch
            entry["speedup_single"] = best["single_s"] / fno_single
            per_query_gain = best["batch_s"] / N_TEST - fno_batch / N_TEST
            entry["break_even_queries"] = (
                (train_s + data_s) / per_query_gain if per_query_gain > 0 else None
            )
        by_grid[str(n)] = entry
        print(f"grid {n}: fno err {fno_err:.2e} {fno_batch*1e3:.1f} ms/batch | matched solver "
              f"{best and best['dt']} {best and best['batch_s']*1e3:.1f} ms/batch", flush=True)

    res = {
        "problem": "periodic viscous Burgers, operator u(.,0) -> u(.,T)",
        "nu": NU, "T": T, "ic_amplitude": AMP, "reference_dt": REF_DT,
        "test_set": f"{N_TEST} unseen random initial conditions (seed {TEST_SEED})",
        "weights_file": f"weights/{WEIGHTS.name}",
        "timing": f"median of repeats after one warm-up, single CPU thread, batch of {N_TEST} "
                  "and single query",
        "fno_train_time_s": train_s,
        "training_data_generation_s": data_s,
        "by_grid": by_grid,
        "system": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(terse=True),
            "processor": platform.processor() or "unknown",
        },
    }
    out = ROOT / "benchmarks" / "speed_burgers_results.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "by_grid"}, indent=2))


if __name__ == "__main__":
    main()
