"""Benchmark datasets and cavity helpers.

Exports the published Ghia et al. (1982) profiles, mesh/boundary-condition helpers,
the divergence-free manufactured solution used for metric verification, and a
centreline comparison utility that turns a cavity solution into Ghia errors.
"""

from .ghia_cavity import (
    AVAILABLE_REYNOLDS,
    GHIA_KNOWN_MISPRINTS,
    GhiaReference,
    available_reynolds,
    cavity_bc_mask,
    cavity_mms_manufactured_solution,
    cavity_mms_streamfunction,
    evaluate_cavity_solution,
    generate_cavity_mesh,
    get_ghia_reference,
    ghia_centerline_errors,
)

__all__ = [
    "AVAILABLE_REYNOLDS",
    "GHIA_KNOWN_MISPRINTS",
    "GhiaReference",
    "available_reynolds",
    "get_ghia_reference",
    "generate_cavity_mesh",
    "cavity_bc_mask",
    "cavity_mms_streamfunction",
    "cavity_mms_manufactured_solution",
    "evaluate_cavity_solution",
    "ghia_centerline_errors",
]
