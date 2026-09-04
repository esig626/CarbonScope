"""Reproducible, non-gating benchmark for cold and reusable native FVA."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path
import platform
import statistics
import time

import highspy
import numpy as np
import pandas as pd

from fluxemu.flux_analysis import (
    compile_flux_lp,
    run_highs_fva_reference,
    run_highs_vffva,
)
from fluxemu.model import FluxModel, load_sbml_flux_model
from fluxemu.real_model import load_ecoli_core_flux_model

CORRECTNESS_TOLERANCE = 1e-7


def measured_times(call: Callable[[], object], repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        values.append(time.perf_counter() - start)
    return values


def measurement(call: Callable[[], object], repeats: int) -> dict[str, object]:
    values = measured_times(call, repeats)
    return {"seconds": values, "median_seconds": statistics.median(values)}


def load_model(argument: str) -> tuple[FluxModel, str, Path | None]:
    if argument == "ecoli-core":
        return load_ecoli_core_flux_model("biomass"), "bundled-e-coli-core-biomass", None
    path = Path(argument)
    return load_sbml_flux_model(path), str(path), path


def compare_ranges(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, object]:
    order_matches = first.index.equals(second.index) and first.columns.equals(second.columns)
    if not order_matches:
        raise RuntimeError("fast FVA result does not preserve reference row/column order")
    difference = float(
        np.max(np.abs(first.to_numpy() - second.to_numpy()), initial=0.0)
    )
    if not np.isfinite(difference) or difference > CORRECTNESS_TOLERANCE:
        raise RuntimeError(
            f"fast FVA differs from the cold reference by {difference:g}, "
            f"above {CORRECTNESS_TOLERANCE:g}"
        )
    return {
        "passed": True,
        "order_matches": True,
        "max_absolute_range_difference": difference,
        "tolerance": CORRECTNESS_TOLERANCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="SBML FBC path, or 'ecoli-core' for the bundled 95-reaction model",
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--parallel-workers", type=int, default=min(2, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--cobra",
        action="store_true",
        help="also benchmark the optional COBRApy processes=1 comparator",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.parallel_workers < 1:
        parser.error("--parallel-workers must be positive")

    model, model_name, sbml_path = load_model(args.model)
    compiled = compile_flux_lp(model)
    parallel_workers = min(args.parallel_workers, 2 * len(compiled.reaction_ids))

    # Correctness is a hard prerequisite; timings are deliberately diagnostic.
    cold = run_highs_fva_reference(model, args.fraction)
    serial = run_highs_vffva(model, args.fraction, workers=1)
    parallel = (
        run_highs_vffva(model, args.fraction, workers=parallel_workers)
        if parallel_workers > 1
        else serial
    )
    serial_correctness = compare_ranges(cold.ranges, serial.ranges)
    parallel_correctness = compare_ranges(cold.ranges, parallel.ranges)

    # Complete one untimed warm-up of each measured path before repetitions.
    run_highs_fva_reference(model, args.fraction)
    run_highs_vffva(model, args.fraction, workers=1)
    if parallel_workers > 1:
        run_highs_vffva(model, args.fraction, workers=parallel_workers)

    measurements = {
        "cold_native_fva": measurement(
            lambda: run_highs_fva_reference(model, args.fraction), args.repeats
        ),
        "fast_native_fva_workers_1": measurement(
            lambda: run_highs_vffva(model, args.fraction, workers=1), args.repeats
        ),
    }
    if parallel_workers > 1:
        measurements[f"fast_native_fva_workers_{parallel_workers}"] = measurement(
            lambda: run_highs_vffva(
                model, args.fraction, workers=parallel_workers
            ),
            args.repeats,
        )

    cobra_version = None
    if args.cobra:
        if sbml_path is None:
            parser.error("--cobra requires an SBML model path")
        try:
            import cobra
            from cobra.io import read_sbml_model

            from fluxemu.cobra_analysis import run_fva
        except ImportError as error:
            parser.error(
                "--cobra requires the optional 'compat' dependencies: " + str(error)
            )
        cobra_model = read_sbml_model(sbml_path)
        run_fva(cobra_model, args.fraction)
        measurements["cobra_fva_processes_1"] = measurement(
            lambda: run_fva(cobra_model, args.fraction), args.repeats
        )
        cobra_version = cobra.__version__

    cold_median = measurements["cold_native_fva"]["median_seconds"]
    serial_median = measurements["fast_native_fva_workers_1"]["median_seconds"]
    speedups = {"fast_serial_over_cold": cold_median / serial_median}
    if parallel_workers > 1:
        key = f"fast_native_fva_workers_{parallel_workers}"
        speedups["fast_parallel_over_cold"] = (
            cold_median / measurements[key]["median_seconds"]
        )

    data = {
        "schema_version": 1,
        "diagnostic_only": True,
        "warmup_runs_per_path": 1,
        "repeats": args.repeats,
        "fraction_of_optimum": args.fraction,
        "model": {
            "name": model_name,
            "fingerprint": compiled.fingerprint,
            "reactions": len(compiled.reaction_ids),
            "balanced_metabolites": len(compiled.balanced_metabolite_ids),
            "nonzeros": compiled.nonzero_count,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "highs": highspy.Highs().version(),
            "cobra": cobra_version,
            "cpu_count": os.cpu_count(),
        },
        "workers": {"serial": 1, "parallel": parallel_workers},
        "correctness": {
            "fast_serial_vs_cold": serial_correctness,
            "fast_parallel_vs_cold": parallel_correctness,
        },
        "measurements": measurements,
        "speedups": speedups,
    }
    print(json.dumps(data, sort_keys=True))


if __name__ == "__main__":
    main()
