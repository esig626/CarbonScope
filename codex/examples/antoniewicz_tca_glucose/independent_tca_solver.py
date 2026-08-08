"""Pure-M+2 reuse of the frozen Antoniewicz direct isotopomer solver.

This module intentionally imports neither FluxEMU nor mfapy.  It loads the
frozen benchmark's NumPy-only 176-state implementation directly and changes
only the acetyl-CoA source distribution for this new tracer experiment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


FROZEN_SOLVER_PATH = Path(__file__).resolve().parents[1] / "antoniewicz_tca" / "direct_isotopomer_solver.py"


def _load_frozen_direct_solver():
    name = "_fluxemu_frozen_antoniewicz_direct_isotopomer_solver"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, FROZEN_SOLVER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - filesystem invariant
        raise ImportError("could not load frozen Antoniewicz direct solver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_FROZEN = _load_frozen_direct_solver()
PURE_M2_ACCOA = _FROZEN.source_distribution(2, {"#11": 1.0})
TCA_METABOLITES = ("OAC", "citrate", "AKG", "glutamate", "succinate", "fumarate")
CARBON_COUNTS = {"OAC": 4, "citrate": 6, "AKG": 5, "glutamate": 5, "succinate": 4, "fumarate": 4}


def solve_stationary(*, symmetric: bool = True) -> dict[str, np.ndarray]:
    """Solve the unchanged full positional TCA system for pure #11 AcCoA."""

    original = _FROZEN.ACCOA
    try:
        _FROZEN.ACCOA = PURE_M2_ACCOA
        return _FROZEN.solve_stationary(symmetric=symmetric)
    finally:
        _FROZEN.ACCOA = original


def mass_isotopomer_distributions(*, symmetric: bool = True) -> dict[str, np.ndarray]:
    """Return complete independent MIDs for all required TCA pools."""

    solution = solve_stationary(symmetric=symmetric)
    return {
        metabolite: _FROZEN.mass_isotopomer_distribution(
            solution[metabolite], CARBON_COUNTS[metabolite]
        )
        for metabolite in TCA_METABOLITES
    }


def isotopomers(carbon_count: int):
    """Expose the ordered states for positional interpretation only."""

    return _FROZEN.isotopomers(carbon_count)


__all__ = [
    "CARBON_COUNTS",
    "PURE_M2_ACCOA",
    "TCA_METABOLITES",
    "isotopomers",
    "mass_isotopomer_distributions",
    "solve_stationary",
]
