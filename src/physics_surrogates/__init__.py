"""Physics Surrogate Benchmarks.

Small, auditable PyTorch implementations of Physics-Informed Neural Networks
(PINNs) and Fourier Neural Operators (FNO), each scored against an analytical
reference solution so the reported error belongs to the model alone.
"""

__version__ = "0.3.0"
__author__ = "Tarik El Moutaouakil"

from . import datasets, models, reference

__all__ = ["models", "datasets", "reference", "__version__"]
