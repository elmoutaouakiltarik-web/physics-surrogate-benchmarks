# Benchmark protocol

The rules every number in this repository follows.

## 1. References: exact where possible, verified where not

| Case | Reference | Where |
|---|---|---|
| Burgers 1D | Cole–Hopf integral, evaluated in log space | `physics_surrogates.reference.cole_hopf_burgers` |
| Heat operator | closed-form Fourier multiplier | `physics_surrogates.models.fno.heat_solution` |
| Cavity | published Ghia et al. (1982) tables | `physics_surrogates.datasets.ghia_cavity` |
| Nonlinear Burgers operator | pseudo-spectral IFRK4 solver, dt = 1e-3 (numerical) | `physics_surrogates.solvers.burgers_solve` |

No model is ever scored against another model, and no reference is produced by the same
solver that produced the prediction.

The Cole–Hopf reference is itself checked two ways: against an independent Fourier
pseudo-spectral solve of the same PDE, and against three exact properties of the problem
(initial condition, Dirichlet boundaries, odd symmetry). See
`tests/test_reference_solutions.py`.

The nonlinear Burgers operator has no closed form for random initial conditions, so its
reference is numerical. `tests/test_solvers.py` verifies it (fourth-order time
convergence, spatial convergence, conservation of the mean, non-growing energy, linear
limit against the exact heat solution); grid-64 and grid-256 references differ by about
2e-3, well below the reported model errors.

## 2. Held-out evaluation

- Burgers: 200 centreline points that are not collocation points, scored at four times.
- FNO (heat and Burgers): 200 initial conditions drawn with seed 1234; training sets use
  seeds 0, 1, 2.
- Cavity: the published probe locations, untouched.

## 3. Raw outputs committed

Each script writes a JSON file in `benchmarks/` containing the configuration, the
hardware/versions and the metrics. README tables are copies of those files. Values such
as point counts are returned by the training code (`BurgersPINN.train`) rather than
re-typed, so the record cannot drift from the run.

## 4. Seeds

The PINN and the FNO heat results are one seeded run per number: treat the third
significant digit as noise. The nonlinear Burgers operator reports mean ± std over 3 seeds
and the per-seed values in `benchmarks/fno_burgers_results.json`.
The FNO training is deterministic on CPU: re-running the published configuration
reproduces the test errors bit-for-bit.

## 5. Weakness is reported, not removed

The PINN headline metric is at t = 0.5 because that is the standard reporting point.
`validation.by_time` records t = 0.25, 0.5, 0.75 and 1.0, which shows the error growing
from 2.5e-2 to 1.5e-1 as the solution steepens. Both appear in the README.

## 6. Speed-ups are accuracy-matched

A surrogate is compared with the solver run at the *same* error, not with the solver
at its finest setting. `scripts/benchmark_speed_burgers.py` sweeps the solver time step,
scores every setting exactly like the FNO, picks the cheapest one that is at least as
accurate, and times both on one CPU thread. Training time and the cost of generating the
training data are reported as a break-even query count, not hidden.

Measured result (one CPU thread): the FNO is 4.2-5.6× faster per query in a batch of 200 and
3.3-4.6× for a single query. The explicit solver turns unstable just above dt = 5e-3 to 6e-3, and
its cheapest stable step is already roughly 260-1,150× more accurate than the FNO, so the "matched"
solver is in practice the cheapest stable one. Training plus data generation (≈170 s) is paid back
after 71,000 (256 points) to 312,000 (64 points) queries. An implicit or adaptive-step solver was
not benchmarked.

## Metrics

- **Relative L2:** `||pred − ref||₂ / ||ref||₂`; for the FNO, averaged over test samples.
- **R²:** `1 − SS_res / SS_tot` over the test points.
- **Max absolute error:** the worst point; for Burgers this is on the shock front.
- **Centerline L2 (cavity):** relative error in percent on the published probe locations,
  via `ghia_centerline_errors()`.

## Reproduce

```bash
python scripts/run_benchmark_real.py        # Burgers PINN (~27 min CPU) + MMS check
python scripts/benchmark_fno_heat.py        # FNO heat operator (~4 min CPU)
python scripts/benchmark_fno_burgers.py     # FNO vs MLP, nonlinear Burgers, 3 seeds (~12 min CPU)
python scripts/benchmark_speed_burgers.py   # accuracy-matched FNO vs solver timing (needs the line above)
python scripts/make_figures.py              # figures from the committed artifacts
pytest -q                                   # includes artifact-consistency checks
```

## Artifact consistency

`tests/test_artifacts.py` enforces the contract between the committed files:

- every figure referenced by the README exists;
- the Burgers JSON contains the configuration actually used and the time profile;
- loading `weights/burgers_pinn.pt` reproduces the published relative L2;
- loading `weights/fno_heat.pt` reproduces the published errors at all three grids;
- loading `weights/fno_burgers.pt` reproduces the seed-0 Burgers errors, and the speed
  benchmark's FNO error is the one recorded for those weights;
- the cavity's shipped profiles are the published ones (minima at the published probe
  locations, 17 probes per table, monotone backflow with Reynolds number).

## Not yet benchmarked

- A lid-driven-cavity surrogate against the Ghia profiles. The data and the comparison
  helper are in place; the model is not.
- FNO on 2D operators (Navier–Stokes, Darcy).
- Multi-seed statistics for the PINN.
- GPU timings.
