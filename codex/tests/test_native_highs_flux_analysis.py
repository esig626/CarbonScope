"""Analytical controls for the cold native HiGHS reference engine."""

from __future__ import annotations

import importlib.util
import math
import sys

import pytest
import fluxemu.flux_analysis.highs as highs_module

from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis import compile_flux_lp, run_highs_fba, run_highs_fva_reference
from fluxemu.model import (FluxMetabolite, FluxModel, FluxReaction, LinearObjective,
                           ObjectiveTerm, StoichiometricTerm)

pytestmark = pytest.mark.skipif(importlib.util.find_spec("highspy") is None,
                                reason="highspy optional extra is unavailable")


def model(*, direction="maximise", objective=(("out", 1.0),), out_bounds=(0, 10),
          balanced=True):
    metabolites = (FluxMetabolite("A", balanced),)
    reactions = (
        FluxReaction("in", (StoichiometricTerm("A", 1),), 0, 10),
        FluxReaction("out", (StoichiometricTerm("A", -1),), *out_bounds),
        FluxReaction("blocked", (), 0, 0),
    )
    return FluxModel(metabolites, reactions,
                     LinearObjective(direction, tuple(ObjectiveTerm(*x) for x in objective)))


def test_compile_is_sparse_ordered_balanced_and_deterministic():
    compiled = compile_flux_lp(model())
    assert compiled.reaction_ids == ("in", "out", "blocked")
    assert compiled.balanced_metabolite_ids == ("A",)
    assert compiled.row_starts == (0, 2)
    assert compiled.column_indices == (0, 1)
    assert compiled.coefficients == (1.0, -1.0)
    assert compiled.nonzero_count == 2
    assert compiled.fingerprint == compile_flux_lp(model()).fingerprint


def test_unique_fba_and_diagnostics():
    result = run_highs_fba(model())
    assert result.objective_value == pytest.approx(10, abs=1e-9)
    assert result.fluxes.to_dict() == pytest.approx({"in": 10, "out": 10, "blocked": 0})
    assert max(result.diagnostics.max_lower_bound_violation,
               result.diagnostics.max_upper_bound_violation,
               result.diagnostics.max_mass_balance_residual,
               result.diagnostics.objective_recalculation_error) <= 1e-9


def test_alternate_optimum_uses_objective_and_fva_invariants_not_incidental_primal():
    alternate = model(balanced=False)
    result = run_highs_fba(alternate)
    assert result.objective_value == pytest.approx(10)
    assert result.diagnostics.max_lower_bound_violation <= 1e-9
    assert result.diagnostics.max_upper_bound_violation <= 1e-9
    assert result.diagnostics.max_mass_balance_residual <= 1e-9
    ranges = run_highs_fva_reference(alternate, 1.0).ranges
    assert ranges.loc["in"].tolist() == pytest.approx([0, 10])


def test_reference_fva_full_and_fractional_ranges_are_repeatable_and_nonmutating():
    original = model(); before = repr(original)
    full = run_highs_fva_reference(original, 1.0)
    assert full.ranges.loc["out"].tolist() == pytest.approx([10, 10])
    assert full.ranges.loc["blocked"].tolist() == pytest.approx([0, 0])
    fractional = run_highs_fva_reference(original, 0.5)
    assert fractional.ranges.loc["out"].tolist() == pytest.approx([5, 10])
    assert fractional.ranges.equals(run_highs_fva_reference(original, 0.5).ranges)
    assert repr(original) == before


def test_reference_fva_has_fresh_single_reaction_objectives_and_retention(monkeypatch):
    calls = []
    original_solve = highs_module._solve

    def recording_solve(lp, costs, direction, operation, retention=None):
        calls.append((tuple(costs), direction, retention))
        return original_solve(lp, costs, direction, operation, retention)

    monkeypatch.setattr(highs_module, "_solve", recording_solve)
    run_highs_fva_reference(model(), 0.5)

    # The first call is biological FBA. Every later cold endpoint contains
    # exactly one fresh cost and the biological objective only as a row.
    assert calls[0] == ((0.0, 1.0, 0.0), "max", None)
    endpoint_calls = calls[1:]
    assert [direction for _, direction, _ in endpoint_calls] == ["min", "max"] * 3
    assert [costs for costs, _, _ in endpoint_calls] == [
        (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
    ]
    assert all(retention == (">=", 5.0) for _, _, retention in endpoint_calls)


def test_reversible_multiobjective_and_minimise_controls():
    reversible = model(out_bounds=(-4, 10), objective=(("in", 1), ("out", 1)))
    assert run_highs_fba(reversible).objective_value == pytest.approx(20)
    minimum = model(direction="minimise", objective=(("out", 1),), out_bounds=(-4, 10))
    assert run_highs_fba(minimum).objective_value == pytest.approx(0)
    assert run_highs_fva_reference(minimum, 1).ranges.loc["out"].tolist() == pytest.approx([0, 0])


@pytest.mark.parametrize("fraction", [0, -1, 1.01, math.nan, True])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(AnalysisError, match="fraction_of_optimum"):
        run_highs_fva_reference(model(), fraction)


def test_infeasible_status_is_explicit():
    impossible = model(out_bounds=(1, 10))
    reactions = (impossible.reactions[0].__class__("in", impossible.reactions[0].stoichiometric_terms, 0, 0),) + impossible.reactions[1:]
    with pytest.raises(AnalysisError, match="FBA.*infeasible"):
        run_highs_fba(FluxModel(impossible.metabolites, reactions, impossible.objective))


def test_malformed_contracts_fail_before_solver():
    with pytest.raises(AnalysisError, match="objective must be nonempty"):
        compile_flux_lp(model(objective=()))
    broken = model(out_bounds=(2, 1))
    with pytest.raises(AnalysisError, match="lower bound exceeds upper"):
        compile_flux_lp(broken)


def test_native_import_firewall_and_lazy_highspy_import():
    source_modules = set(sys.modules)
    assert "cobra" not in __import__("fluxemu.flux_analysis.highs", fromlist=["x"]).__dict__
    source = __import__("inspect").getsource(__import__("fluxemu.flux_analysis.highs", fromlist=["x"]))
    assert all(token not in source for token in ("import cobra", "import optlang", "import mfapy", "scipy.optimize"))
    assert "highspy" not in source_modules or importlib.util.find_spec("highspy") is not None
