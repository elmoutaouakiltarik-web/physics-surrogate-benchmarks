#!/usr/bin/env python3
"""FNO1d on the periodic heat equation: learn u(., 0) -> u(., T) against the exact FFT solution.

Writes benchmarks/fno_heat_results.json. CPU, a few minutes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physics_surrogates.models.fno import (  # noqa: E402
    evaluate,
    random_periodic_fields,
    train_fno_heat,
)

NU, T, N_TRAIN, N_GRID, EPOCHS, SEED = 0.01, 1.0, 1000, 64, 300, 0


def main() -> None:
    t0 = time.perf_counter()
    model, history = train_fno_heat(
        n_train=N_TRAIN, n_grid=N_GRID, nu=NU, t=T, epochs=EPOCHS, seed=SEED
    )
    train_s = time.perf_counter() - t0
    test = {n: evaluate(model, random_periodic_fields(200, n, seed=1234), NU, T) for n in (64, 128, 256)}
    res = {
        "problem": "periodic heat equation u_t = nu u_xx on [0,1), operator u(.,0) -> u(.,T)",
        "nu": NU, "T": T, "train_samples": N_TRAIN, "train_grid": N_GRID,
        "epochs": EPOCHS, "seed": SEED,
        "params": sum(p.numel() for p in model.parameters()),
        "train_time_s": round(train_s, 1),
        "final_train_rel_l2": history[-1],
        "test_rel_l2_by_grid": {str(k): v for k, v in test.items()},
        "reference": "exact Fourier-space solution (no solver error)",
        "test_set": "200 unseen random initial conditions (seed 1234)",
    }
    out = ROOT / "benchmarks" / "fno_heat_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
