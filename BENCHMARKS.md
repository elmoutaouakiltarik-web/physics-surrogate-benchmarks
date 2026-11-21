# Benchmark protocol

The rules every number in this repository follows.

## 1. Exact references only

| Case | Reference | Where |
|---|---|---|
| Burgers 1D | Cole–Hopf integral, evaluated in log space | `physics_surrogates.reference.cole_hopf_burgers` |
| Heat operator | closed-form Fourier multiplier | `physics_surrogates.models.fno.heat_solution` |
| Cavity | published Ghia et al. (1982) tables | `physics_surrogates.datasets.ghia_cavity` |

No model is ever scored against another model, and no reference is produced by the same
solver that produced the prediction.

The Cole–Hopf reference is itself checked two ways: against an independent Fourier
pseudo-spectral solve of the same PDE, and against three exact properties of the problem
(initial condition, Dirichlet boundaries, odd symmetry). See
`tests/test_reference_solutions.py`.

## 2. Held-out evaluation

- Burgers: 200 centreline points that are not collocation points, scored at four times.
- FNO: 200 initial conditions drawn with seed 1234; the training set used seed 0.
- Cavity: the published probe locations, untouched.

## 3. Raw outputs committed

Each script writes a JSON file in `benchmarks/` containing the configuration, the
hardware/versions and the metrics. README tables are copies of those files. Values such
as point counts are returned by the training code (`BurgersPINN.train`) rather than
re-typed, so the record cannot drift from the run.

## 4. One seeded run per number

There are no multi-seed statistics yet. Treat the third significant digit as noise.
The FNO training is deterministic on CPU: re-running the published configuration
reproduces the test errors bit-for-bit.

## 5. Weakness is reported, not removed

The PINN headline metric is at t = 0.5 because that is the standard reporting point.
`validation.by_time` records t = 0.25, 0.5, 0.75 and 1.0, which shows the error growing
from 2.5e-2 to 1.5e-1 as the solution steepens. Both appear in the README.

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
python scripts/make_figures.py              # figures from the committed artifacts
pytest -q                                   # includes artifact-consistency checks
```

## Artifact consistency

`tests/test_artifacts.py` enforces the contract between the committed files:

- every figure referenced by the README exists;
- the Burgers JSON contains the configuration actually used and the time profile;
- loading `weights/burgers_pinn.pt` reproduces the published relative L2;
- loading `weights/fno_heat.pt` reproduces the published errors at all three grids;
- the cavity's shipped profiles are the published ones (minima at the published probe
  locations, 17 probes per table, monotone backflow with Reynolds number).

## Not yet benchmarked

- A lid-driven-cavity surrogate against the Ghia profiles. The data and the comparison
  helper are in place; the model is not.
- FNO on a nonlinear operator (Burgers, Navier–Stokes).
- Multi-seed statistics and error bars.
- GPU timings.
