# Contributing

Thanks for taking the time to improve this repository. It is a small benchmark
project, so the bar is less about volume of code and more about whether a number
can be trusted.

## The one rule that matters

**Every reported number must come from a run whose configuration is recorded.**

Concretely:

- A metric lives in `benchmarks/*.json` first. The README table is a copy of it.
- Configuration values in the JSON are produced by the training code, never typed
  by hand (see `BurgersPINN.train`, which returns the point counts it actually used).
- If you change a model or a dataset, re-run the benchmark and commit the new JSON
  together with the code change. Do not leave a README table describing a previous run.
- If a result is a smoke test, a synthetic check or a known-weak case, label it.

`tests/test_artifacts.py` enforces the mechanical part of this: it reloads the
committed weights, recomputes the published error and fails if the JSON and the
weights disagree.

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/physics-surrogate-benchmarks.git
cd physics-surrogate-benchmarks
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # skip if you have CUDA
pip install -e ".[dev]"
pre-commit install                                  # optional but recommended
pytest -q
```

## Before opening a pull request

```bash
ruff check src tests scripts     # lint
pytest tests/ -q                 # unit + artifact-consistency tests
```

If you touched a benchmark, also run the entry point you changed and include the
regenerated JSON (and the figure, if it depends on the changed artifact):

```bash
python scripts/run_benchmark_real.py        # ~27 min CPU, Burgers PINN
python scripts/benchmark_fno_heat.py        # ~4 min CPU, FNO heat operator
python scripts/make_figures.py              # figures from committed artifacts
```

## Contribution guidelines

### Models

- Keep the interface (`train`, `predict`, `save_weights`, `load_weights`) familiar.
- Document the equation, domain, initial and boundary conditions in the module docstring.
- Add a test for the forward/backward path and at least one exact property
  (boundary condition, symmetry, divergence, convergence) rather than only shapes.

### Reference solutions

A reference must be independent of the model being scored. If you add one, add a
cross-check against a second method, as `tests/test_reference_solutions.py` does
for the Cole-Hopf quadrature versus a pseudo-spectral solve.

### Datasets

- Cite the published source and keep the published probe locations.
- Transcribe values, do not smooth or interpolate them.
- Record known misprints in the data module instead of silently correcting them.
- Never invent reference data to fill a missing Reynolds number; raise instead.

### Figures

`docs/img/*.png` are generated, not hand-edited. `scripts/make_figures.py` owns the
model and reference figures; `scripts/run_benchmark_real.py --mms-only` owns the MMS
verification figure. A figure that cannot be regenerated from committed artifacts
should not be in the README.

## Commit style

Conventional commits, one logical change per commit:

```
feat(fno): report the epoch-mean training loss
fix(ghia): correct the Re=1000 centreline tables
docs: rewrite the reproduction section
test: cross-check Cole-Hopf against a spectral solver
ci: verify the wheel contains every subpackage
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`.

## Pull request checklist

- [ ] `ruff check src tests scripts` passes
- [ ] `pytest tests/ -q` passes
- [ ] Benchmark JSON regenerated if a model or dataset changed
- [ ] README numbers still match `benchmarks/*.json`
- [ ] No secrets, credentials, personal paths or proprietary code
- [ ] New claims about a metric are reproducible from a command in the README

## Reporting issues

Include what you ran, the Python/PyTorch versions, the platform, and the full error.
For a numerical disagreement, say which reference and which tolerance you expected.

Questions and proposals are welcome in GitHub Issues; benchmarks are often more
useful when someone points out that a comparison is too easy.
