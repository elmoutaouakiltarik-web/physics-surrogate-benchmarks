# Changelog

## 0.2.0 — 2026-11-21

### Corrected

- **Ghia et al. (1982) reference data.** The shipped arrays did not match the paper:
  the `v` profiles reused the Table I `y` grid instead of the Table II `x` grid, and the
  `u` values below y ≈ 0.05 implied a minimum of −0.338 at Re = 100 where the published
  minimum is −0.211. All three Reynolds numbers were re-transcribed from the published
  tables (17 probes per table) and are now covered by tests that check the values at the
  published minima and extrema.
- **Result files recorded the wrong training configuration.** The benchmark trained with
  256 initial-condition and 256 boundary points but wrote 100/100 into the JSON. Those
  values, together with the point count after RAR, are now returned by the training code.
- **Project URLs** in `pyproject.toml` pointed at a non-existent account
  (`tarik-el-m`), so the package metadata linked to a dead repository.
- **Packaging** declared only the top-level package, so a built wheel was missing
  `physics_surrogates.models` and `physics_surrogates.datasets`, and referenced a
  `py.typed` file that did not exist.
- **`scripts/benchmark_cavity_burgers.py`** was a second copy of the Burgers benchmark
  that overwrote the published JSON with a schema that had no validation block, and
  forced `device="cuda"` unconditionally. It is now a shim over the canonical script.
- **FNO training loss** was the loss of the last mini-batch; it is now the epoch mean.

### Added

- `physics_surrogates.reference` with the Cole–Hopf solution and an independent
  pseudo-spectral Burgers solver, cross-checked in `tests/test_reference_solutions.py`.
- `validation.by_time` in the Burgers result file (t = 0.25, 0.5, 0.75, 1.0), which
  makes the degradation towards t = 1 explicit.
- Committed FNO weights (`weights/fno_heat.pt`) and weight save/load helpers.
- `scripts/make_figures.py`: every figure is regenerated from committed artifacts.
- `ghia_centerline_errors()`, `available_reynolds()`, a stream-function helper and
  shape validation in `evaluate_cavity_solution()`.
- `tests/test_artifacts.py`: README figures, JSON schema and weight/metric agreement.
- CI jobs for the built wheel and for end-to-end benchmark scripts.
- `--from-weights`, `--mms-only` and `--quick` modes for the Burgers benchmark.

### Removed

- `torch[cuda11]` optional dependency (that extra no longer exists).

## 0.1.0 — 2024-11-15

- Initial PINN (Burgers) and FNO (heat) benchmarks with committed results.
