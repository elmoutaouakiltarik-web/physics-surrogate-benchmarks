"""Deprecated alias - kept so older instructions keep working.

This file used to hold a second, divergent copy of the Burgers benchmark that
wrote ``benchmarks/burgers_1d_results.json`` *without* the validation block,
silently destroying the published numbers. It now just forwards to the single
canonical entry point, ``scripts/run_benchmark_real.py``.

    python scripts/benchmark_cavity_burgers.py            # same as run_benchmark_real
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("run_benchmark_real.py")
    print(f"[deprecated] use {target.name}; forwarding to it", file=sys.stderr)
    runpy.run_path(str(target), run_name="__main__")
