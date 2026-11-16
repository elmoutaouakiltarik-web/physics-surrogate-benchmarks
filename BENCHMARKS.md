# Benchmark protocol

Rules every number in this repository follows:

1. **Exact references only.** Burgers is compared with the Cole–Hopf integral solution (`scripts/run_benchmark_real.py`, `cole_hopf_burgers`); the heat operator with its closed-form Fourier solution (`models/fno.py`, `heat_solution`). No model is scored against another model.
2. **Held-out evaluation.** Burgers: 200 test points not used as collocation points. FNO: 200 initial conditions drawn with a different seed (1234) from the training set (seed 0).
3. **Raw outputs committed.** Each script writes a JSON file in `benchmarks/` with configuration, hardware and metrics. README tables are copied from those files.
4. **One seeded run per number.** No multi-seed statistics yet; treat third significant digits as noise.

## Metrics

- Relative L2: `||pred - ref||_2 / ||ref||_2` (FNO: averaged over test samples).
- R²: `1 - SS_res / SS_tot` over the test points.
- Max absolute error: worst point; for Burgers this is on the shock front.

## Reproduce

```bash
python scripts/run_benchmark_real.py   # Burgers PINN, ~27 min on a laptop CPU
python scripts/benchmark_fno_heat.py   # FNO heat operator, ~4 min on a laptop CPU
```

## Not yet benchmarked

- Lid-driven cavity surrogate against the Ghia et al. (1982) profiles shipped in `datasets/ghia_cavity.py`.
- FNO on a nonlinear operator (Burgers, Navier–Stokes).
- `benchmarks/mms_cavity_results.json` is a metric-pipeline check on a manufactured solution with synthetic noise, not a model result.
