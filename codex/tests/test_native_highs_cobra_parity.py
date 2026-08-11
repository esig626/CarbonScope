"""Strict COBRA 0.31.1 parity over repository models."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest
from cobra.io import read_sbml_model
from fluxemu.cobra_analysis import run_fba, run_fva
from fluxemu.compat import project_cobra_flux_model
from fluxemu.flux_analysis import run_highs_fba, run_highs_fva_reference

pytestmark = pytest.mark.skipif(importlib.util.find_spec("highspy") is None,
                                reason="highspy optional extra is unavailable")
ROOT = Path(__file__).parents[1]
MODELS = [ROOT / "examples/toy_model.xml",
          ROOT / "examples/antoniewicz_tca/antoniewicz_tca.xml",
          ROOT / "examples/antoniewicz_tca_glucose/antoniewicz_tca_glucose.xml"]
PARITY_TOLERANCE = 1e-7

@pytest.mark.parametrize("path", MODELS, ids=lambda x: x.stem)
def test_native_fba_and_fva_match_cobra(path):
    cobra = read_sbml_model(path); canonical = project_cobra_flux_model(cobra)
    cobra_fba, native_fba = run_fba(cobra), run_highs_fba(canonical)
    assert native_fba.objective_direction == cobra_fba.objective_direction
    assert abs(native_fba.objective_value - cobra_fba.objective_value) <= PARITY_TOLERANCE
    assert native_fba.diagnostics.max_lower_bound_violation <= PARITY_TOLERANCE
    assert native_fba.diagnostics.max_upper_bound_violation <= PARITY_TOLERANCE
    assert native_fba.diagnostics.max_mass_balance_residual <= PARITY_TOLERANCE
    cobra_fva, native_fva = run_fva(cobra, 1.0), run_highs_fva_reference(canonical, 1.0)
    assert tuple(native_fva.ranges.index) == tuple(reaction.id for reaction in cobra.reactions)
    differences = (native_fva.ranges - cobra_fva.ranges).abs()
    worst_min = differences["minimum"].idxmax(); worst_max = differences["maximum"].idxmax()
    assert differences.loc[worst_min, "minimum"] <= PARITY_TOLERANCE, (worst_min, differences.loc[worst_min, "minimum"])
    assert differences.loc[worst_max, "maximum"] <= PARITY_TOLERANCE, (worst_max, differences.loc[worst_max, "maximum"])
