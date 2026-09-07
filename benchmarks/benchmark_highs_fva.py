"""Diagnostic native FVA benchmark: cold reference versus reusable FastFVA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import highspy
import numpy as np

from fluxemu.flux_analysis import run_highs_fva_reference, run_highs_vffva
from fluxemu.model import load_sbml_flux_model
from fluxemu.real_model import load_ecoli_core_flux_model


TOLERANCE = 1e-7


def _times(call, repeats: int) -> list[float]:
    result = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        result.append(time.perf_counter() - started)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="SBML FBC path or 'ecoli-core'")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1:
        parser.error("--repeats and --workers must be positive")

    model = (
        load_ecoli_core_flux_model("biomass")
        if args.model == "ecoli-core"
        else load_sbml_flux_model(Path(args.model))
    )
    cold = run_highs_fva_reference(model, args.fraction)
    fast = run_highs_vffva(model, args.fraction, workers=args.workers)
    if not cold.ranges.index.equals(fast.ranges.index):
        raise RuntimeError("FastFVA changed canonical reaction order")
    discrepancy = float(np.max(np.abs(cold.ranges.to_numpy() - fast.ranges.to_numpy()), initial=0.0))
    if not np.isfinite(discrepancy) or discrepancy > TOLERANCE:
        raise RuntimeError(f"FastFVA/reference discrepancy {discrepancy:g} exceeds {TOLERANCE:g}")

    run_highs_fva_reference(model, args.fraction)
    run_highs_vffva(model, args.fraction, workers=args.workers)
    cold_times = _times(lambda: run_highs_fva_reference(model, args.fraction), args.repeats)
    fast_times = _times(lambda: run_highs_vffva(model, args.fraction, workers=args.workers), args.repeats)
    output = {
        "diagnostic_only": True,
        "python": platform.python_version(),
        "highs": highspy.Highs().version(),
        "fraction_of_optimum": args.fraction,
        "workers": args.workers,
        "repeats": args.repeats,
        "max_absolute_range_difference": discrepancy,
        "cold_seconds": cold_times,
        "fast_seconds": fast_times,
        "cold_median_seconds": statistics.median(cold_times),
        "fast_median_seconds": statistics.median(fast_times),
        "speedup": statistics.median(cold_times) / statistics.median(fast_times),
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
