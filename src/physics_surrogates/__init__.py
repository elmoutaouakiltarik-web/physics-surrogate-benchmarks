"""
Physics Surrogate Benchmarks

Reproducible benchmarks for Physics-Informed Neural Networks (PINNs) and
Fourier Neural Operators (FNO) on classical CFD problems.
"""

__version__ = "0.1.0"
__author__ = "Tarik El Moutaouakil"

from . import datasets, models

__all__ = ["models", "datasets"]
