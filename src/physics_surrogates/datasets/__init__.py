"""Benchmark datasets: Ghia cavity, MMS verification."""

from .ghia_cavity import (
    GhiaReference,
    cavity_bc_mask,
    cavity_mms_manufactured_solution,
    evaluate_cavity_solution,
    generate_cavity_mesh,
    get_ghia_reference,
)

__all__ = [
    "GhiaReference",
    "get_ghia_reference",
    "generate_cavity_mesh",
    "cavity_bc_mask",
    "cavity_mms_manufactured_solution",
    "evaluate_cavity_solution",
]
