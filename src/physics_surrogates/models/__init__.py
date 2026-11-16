"""Surrogate models: PINN (Burgers 1D) and FNO (1D operator learning)."""

from .fno import FNO1d, SpectralConv1d
from .pinns import (
    BurgersPINN,
    PINNNet,
    RARSampler,
    SinActivation,
)

__all__ = [
    "FNO1d",
    "SpectralConv1d",
    "PINNNet",
    "SinActivation",
    "BurgersPINN",
    "RARSampler",
]
