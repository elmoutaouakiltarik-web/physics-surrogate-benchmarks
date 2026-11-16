# Contributing to Physics Surrogate Benchmarks

Thank you for your interest in contributing! This document outlines the process.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/physics-surrogate-benchmarks.git
   cd physics-surrogate-benchmarks
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We follow PEP 8 with the following tools:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type hints
mypy src/physics_surrogates --ignore-missing-imports
```

### Writing Tests

All new features must include tests in `tests/`:

```python
import pytest
from physics_surrogates.models import BurgersPINN

def test_burgers_train():
    """Test PINN training."""
    pinn = BurgersPINN(device="cpu")
    metrics = pinn.train(epochs_adam=10, epochs_lbfgs=1)
    assert "final_loss" in metrics
    assert metrics["final_loss"] > 0
```

Run tests locally:

```bash
pytest tests/ -v --cov=src/physics_surrogates
```

### Committing Changes

Use conventional commits:

```
feat(pinns): add RAR sampling to Burgers PINN
fix(ghia): correct Ghia reference data for Re=400
docs: update cavity benchmark README
test: add MMS verification tests
```

Format: `<type>(<scope>): <message>`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`

### Pull Request Process

1. **Push your branch** to your fork
2. **Create a PR** with a clear title and description
3. **Link any related issues**: "Fixes #123"
4. **Wait for CI/CD** — all tests must pass
5. **Address review comments**
6. **Merge** once approved

## Contribution Guidelines

### Models

- Follow the PINN/FNO/DeepONet interface
- Include docstrings with math notation
- Add unit tests for forward/backward pass
- Provide a minimal example in docstring

### Datasets

- Reference published benchmarks (Ghia, NASA, etc.)
- Include data loading + validation
- Test against known values (R², RMSE)
- Document domain, BC, IC clearly

### Benchmarks

- Report wall-clock time on standard hardware
- Compare multiple models fairly
- Include reproducibility info (seed, device, version)
- Plot results with matplotlib

## Reporting Issues

Use GitHub Issues for:
- **Bug reports**: Include Python version, PyTorch version, error message
- **Feature requests**: Explain motivation and use case
- **Questions**: Check closed issues first

Template:

```markdown
## Bug Report

**Description:**
Brief description of the issue.

**Steps to reproduce:**
1. ...
2. ...

**Expected behavior:**
What should happen.

**Actual behavior:**
What actually happens.

**Environment:**
- Python: 3.10
- PyTorch: 2.0
- OS: Ubuntu 22.04
- Device: CUDA 11.8
```

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows `black` + `ruff` style
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Type hints added where applicable
- [ ] Docstrings present (models especially)
- [ ] No secrets (keys, paths, passwords)
- [ ] Benchmarks run and report realistic metrics
- [ ] README/docs updated if needed

## Setting Up Pre-Commit Hooks

Optional but recommended:

```bash
pip install pre-commit
pre-commit install
```

Then configure `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.0.260
    hooks:
      - id: ruff
```

## Questions?

- Open a GitHub Discussion
- Email: tarik@example.com
- Check existing issues first

---

**Thank you for contributing!** Every improvement helps the community.
