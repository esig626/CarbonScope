"""Analytical controls for the cold native HiGHS reference engine."""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest
import fluxemu.flux_analysis.highs as highs_module

from fluxemu.exceptions import AnalysisError
from fluxemu.flux_analysis import (
    compile_flux_lp,
    prepare_highs_flux_region,
    run_highs_fba,
    run_highs_fva_reference,
)
from fluxemu.model import (FluxMetabolite, FluxModel, FluxReaction, LinearObjective,
                           ObjectiveTerm, StoichiometricTerm)

pytestmark = pytest.mark.skipif(importlib.util.find_spec("highspy") is None,
                                reason="highspy is unavailable")


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


@pytest.mark.parametrize(
    ("direction", "tiny_coefficient"),
    (("maximise", -1e-12), ("minimise", 1e-12)),
)
def test_compile_fails_closed_on_material_mass_coefficient_below_highs_resolution(
    direction, tiny_coefficient
):
    unresolved = FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("M", 1.0),), -0.1, 0.1
            ),
            FluxReaction(
                "Y",
                (StoichiometricTerm("M", tiny_coefficient),),
                0.0,
                1e12,
            ),
        ),
        LinearObjective(direction, (ObjectiveTerm("X", 1.0),)),
    )

    with pytest.raises(AnalysisError) as caught:
        compile_flux_lp(unresolved)

    message = str(caught.value).lower()
    assert "coefficient" in message
    assert "highs" in message
    assert "resolution" in message or "represent" in message
    assert "mass" in message or "equality" in message


@pytest.mark.parametrize("tiny_coefficient", [-1e-12, 1e-12])
def test_compile_accepts_subresolution_mass_coefficient_with_tolerable_span(
    tiny_coefficient,
):
    tolerable = FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("M", 1.0),), -0.1, 0.1
            ),
            FluxReaction(
                "Y",
                (StoichiometricTerm("M", tiny_coefficient),),
                0.0,
                1e4,
            ),
        ),
        LinearObjective("maximise", (ObjectiveTerm("X", 1.0),)),
    )

    compiled = compile_flux_lp(tolerable)

    assert compiled.solver_coefficients == pytest.approx(
        (1.0, tiny_coefficient)
    )
    assert abs(tiny_coefficient) * 1e4 <= highs_module.FEASIBILITY_TOLERANCE


@pytest.mark.parametrize(
    ("direction", "tiny_coefficient"),
    (("maximise", -5e-13), ("minimise", 5e-13)),
)
def test_compile_fails_closed_on_aggregate_subresolution_mass_activity(
    direction, tiny_coefficient
):
    aggregate = FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("M", 1.0),), -0.1, 0.1
            ),
            *tuple(
                FluxReaction(
                    f"Y{index}",
                    (StoichiometricTerm("M", tiny_coefficient),),
                    0.0,
                    1e5,
                )
                for index in range(3)
            ),
        ),
        LinearObjective(direction, (ObjectiveTerm("X", 1.0),)),
    )

    with pytest.raises(AnalysisError) as caught:
        compile_flux_lp(aggregate)

    message = str(caught.value).lower()
    assert all(token in message for token in ("coefficient", "highs", "aggregate"))
    assert "resolution" in message or "represent" in message
    assert "mass" in message or "equality" in message


@pytest.mark.parametrize("tiny_coefficient", [-5e-13, 5e-13])
def test_compile_accepts_tolerable_aggregate_subresolution_mass_activity(
    tiny_coefficient,
):
    aggregate = FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("M", 1.0),), -0.1, 0.1
            ),
            *tuple(
                FluxReaction(
                    f"Y{index}",
                    (StoichiometricTerm("M", tiny_coefficient),),
                    0.0,
                    5e4,
                )
                for index in range(3)
            ),
        ),
        LinearObjective("maximise", (ObjectiveTerm("X", 1.0),)),
    )

    compiled = compile_flux_lp(aggregate)

    assert compiled.solver_coefficients == pytest.approx(
        (1.0, tiny_coefficient, tiny_coefficient, tiny_coefficient)
    )
    assert 3.0 * abs(tiny_coefficient) * 5e4 <= highs_module.FEASIBILITY_TOLERANCE


@pytest.mark.parametrize(
    ("direction", "resolved_coefficient", "expected_x"),
    (("maximise", -2e-12, 2e-7), ("minimise", 2e-12, -2e-7)),
)
def test_fba_solves_mass_coefficient_above_highs_matrix_resolution(
    direction, resolved_coefficient, expected_x
):
    resolved = FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("M", 1.0),), -0.1, 0.1
            ),
            FluxReaction(
                "Y",
                (StoichiometricTerm("M", resolved_coefficient),),
                0.0,
                1e5,
            ),
        ),
        LinearObjective(direction, (ObjectiveTerm("X", 1.0),)),
    )

    result = run_highs_fba(resolved)

    assert result.objective_value == pytest.approx(expected_x)
    assert result.fluxes["X"] == pytest.approx(expected_x)
    assert result.fluxes["Y"] == pytest.approx(1e5)
    assert result.diagnostics.max_mass_balance_residual <= 1e-12


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

    # The first call is biological FBA using the normalized exact quotient:
    # canonical elimination keeps out as the free coordinate under in == out.
    # Every later cold endpoint contains exactly one fresh cost and retains the
    # biological objective only as a conditioned row.
    assert calls[0] == ((0.0, 1.0, 0.0), "max", None)
    endpoint_calls = calls[1:]
    assert [direction for _, direction, _ in endpoint_calls] == ["min", "max"] * 3
    assert [costs for costs, _, _ in endpoint_calls] == [
        (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
    ]
    assert all(
        retention is not None
        and retention.sense == ">="
        and retention.bound == pytest.approx(5.0)
        and retention.effective_bound == pytest.approx(5.0)
        for _, _, retention in endpoint_calls
    )


def test_reversible_multiobjective_and_minimise_controls():
    reversible = model(out_bounds=(-4, 10), objective=(("in", 1), ("out", 1)))
    assert run_highs_fba(reversible).objective_value == pytest.approx(20)
    minimum = model(direction="minimise", objective=(("out", 1),), out_bounds=(-4, 10))
    assert run_highs_fba(minimum).objective_value == pytest.approx(0)
    assert run_highs_fva_reference(minimum, 1).ranges.loc["out"].tolist() == pytest.approx([0, 0])


@pytest.mark.parametrize(
    ("direction", "sign"),
    (
        ("maximise", 1.0),
        ("minimise", -1.0),
    ),
)
def test_fba_fails_closed_below_highs_dual_objective_resolution(direction, sign):
    wide = FluxModel(
        (),
        (
            FluxReaction("X", (), 0.0, 1e-12),
            FluxReaction("Y", (), 0.0, 1e12),
        ),
        LinearObjective(
            direction,
            (
                ObjectiveTerm("X", sign),
                ObjectiveTerm("Y", sign * 1e-10),
            ),
        ),
    )

    with pytest.raises(AnalysisError) as caught:
        run_highs_fba(wide)

    message = str(caught.value).lower()
    assert all(token in message for token in ("objective", "coefficient", "highs"))
    assert "resolution" in message or "represent" in message


@pytest.mark.parametrize(
    ("direction", "sign"),
    (("maximise", 1.0), ("minimise", -1.0)),
)
def test_public_fba_rejects_any_subdual_term_in_exact_optimum(direction, sign):
    model_with_unresolved_optimum = FluxModel(
        (),
        (
            FluxReaction("X", (), -1.0, 1.0),
            FluxReaction("Z", (), -1.0, 1.0),
        ),
        LinearObjective(
            direction,
            (
                ObjectiveTerm("X", sign * 1e-10),
                ObjectiveTerm("Z", sign * 10.0),
            ),
        ),
    )

    with pytest.raises(AnalysisError) as caught:
        run_highs_fba(model_with_unresolved_optimum)

    message = str(caught.value).lower()
    assert all(
        token in message
        for token in ("exact", "optimal", "face", "highs", "resolution")
    )
    assert "reaction='x'" in message


@pytest.mark.parametrize(
    ("direction", "sign"),
    (("maximise", 1.0), ("minimise", -1.0)),
)
def test_fba_fails_closed_on_aggregate_subresolution_objective_activity(
    direction, sign
):
    aggregate = FluxModel(
        (),
        (
            FluxReaction("X", (), 0.0, 1.0),
            *tuple(
                FluxReaction(f"Y{index}", (), 0.0, 5e4)
                for index in range(3)
            ),
        ),
        LinearObjective(
            direction,
            (
                ObjectiveTerm("X", sign),
                *tuple(
                    ObjectiveTerm(f"Y{index}", sign * 1e-12)
                    for index in range(3)
                ),
            ),
        ),
    )

    with pytest.raises(AnalysisError) as caught:
        run_highs_fba(aggregate)

    message = str(caught.value).lower()
    assert all(
        token in message
        for token in ("objective", "coefficient", "highs", "aggregate")
    )
    assert "resolution" in message or "represent" in message


@pytest.mark.parametrize(
    ("direction", "sign", "expected_objective"),
    (("maximise", 1.0, 1.0), ("minimise", -1.0, -1.0)),
)
def test_fractional_preparation_allows_tolerable_aggregate_objective_activity(
    direction, sign, expected_objective
):
    aggregate = FluxModel(
        (),
        (
            FluxReaction("X", (), 0.0, 1.0),
            *tuple(
                FluxReaction(f"Y{index}", (), 0.0, 3e4)
                for index in range(3)
            ),
        ),
        LinearObjective(
            direction,
            (
                ObjectiveTerm("X", sign),
                *tuple(
                    ObjectiveTerm(f"Y{index}", sign * 1e-12)
                    for index in range(3)
                ),
            ),
        ),
    )

    prepared = prepare_highs_flux_region(aggregate, 0.8)

    assert prepared.fba.objective_value == pytest.approx(
        expected_objective,
        abs=highs_module.OBJECTIVE_TOLERANCE,
        rel=0.0,
    )
    assert prepared.fba.fluxes["X"] == pytest.approx(1.0)
    assert prepared.retention.bound == pytest.approx(0.8 * expected_objective)
    assert 3.0 * 1e-12 * 3e4 <= highs_module.OBJECTIVE_TOLERANCE


@pytest.mark.parametrize(
    ("direction", "sign", "expected_objective"),
    (
        ("maximise", 1.0, 100.000000000001),
        ("minimise", -1.0, -100.000000000001),
    ),
)
def test_fba_solves_material_wide_span_above_highs_dual_resolution(
    direction, sign, expected_objective
):
    wide = FluxModel(
        (),
        (
            FluxReaction("X", (), 0.0, 1e-12),
            FluxReaction("Y", (), 0.0, 5e10),
        ),
        LinearObjective(
            direction,
            (
                ObjectiveTerm("X", sign),
                ObjectiveTerm("Y", sign * 2e-9),
            ),
        ),
    )

    result = run_highs_fba(wide)

    assert result.objective_value == pytest.approx(expected_objective)
    assert result.fluxes["X"] == pytest.approx(1e-12)
    assert result.fluxes["Y"] == pytest.approx(5e10)


def test_fba_fails_closed_when_material_objective_is_below_solver_resolution():
    unrepresentable = FluxModel(
        (),
        (
            FluxReaction("X", (), 0.0, 1e-12),
            FluxReaction("Y", (), 0.0, 1e15),
        ),
        LinearObjective(
            "maximise",
            (
                ObjectiveTerm("X", 1.0),
                ObjectiveTerm("Y", 1e-13),
            ),
        ),
    )

    with pytest.raises(AnalysisError) as caught:
        run_highs_fba(unrepresentable)

    message = str(caught.value).lower()
    assert "objective" in message
    assert "coefficient" in message
    assert "solver" in message or "highs" in message
    assert "resolution" in message or "represent" in message


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


def test_reduced_objective_helper_rejects_material_fixed_mass_inconsistency():
    reduced = FluxModel(
        (FluxMetabolite("A", True),),
        (
            FluxReaction("X", (StoichiometricTerm("A", 1.0),), -1.0, 1.0),
            FluxReaction("Y", (StoichiometricTerm("A", 1.0),), -1.0, 1.0),
            FluxReaction("Z", (), 0.0, 10.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("Z", 1.0),)),
    )
    lp = compile_flux_lp(reduced)

    with pytest.raises(
        AnalysisError,
        match="fixed-coordinate substitution.*inconsistent",
    ):
        highs_module._condition_objective_after_fixed_coordinates(
            lp, (0, 1), (1.0, 1.0)
        )

    effective, constant, scale = (
        highs_module._condition_objective_after_fixed_coordinates(
            lp, (0, 1), (1.0, -1.0 + 5e-8)
        )
    )
    assert effective == pytest.approx((0.0, 0.0, 1.0))
    assert constant == pytest.approx(0.0)
    assert scale == pytest.approx(1.0)


def test_reduced_objective_helper_preserves_ulp_anchor_but_rejects_bad_bounds():
    bounded = FluxModel(
        (),
        (
            FluxReaction("X", (), -1.0, 1.0),
            FluxReaction("Q", (), 2.0, 2.0),
        ),
        LinearObjective(
            "maximise",
            (ObjectiveTerm("X", 1.0), ObjectiveTerm("Q", 1.0)),
        ),
    )
    lp = compile_flux_lp(bounded)

    effective, constant, scale = (
        highs_module._condition_objective_after_fixed_coordinates(
            lp,
            (0,),
            (-1.0 - 5e-8,),
        )
    )
    assert effective == pytest.approx((0.0, 0.0))
    assert constant == pytest.approx(1.0 - 5e-8)
    assert scale == pytest.approx(0.0)

    effective, constant, scale = (
        highs_module._condition_objective_after_fixed_coordinates(
            lp,
            (1,),
            (2.0 + 5e-8,),
        )
    )
    assert effective == pytest.approx((1.0, 0.0))
    assert constant == pytest.approx(2.0 + 5e-8)
    assert scale == pytest.approx(1.0)

    for index, value in ((0, -1.0 - 1e-6), (1, 2.0 + 1e-6)):
        with pytest.raises(
            AnalysisError,
            match="fixed objective coordinate.*reaction bounds",
        ):
            highs_module._condition_objective_after_fixed_coordinates(
                lp,
                (index,),
                (value,),
            )


@pytest.mark.parametrize("kind", ["stoichiometry", "objective"])
def test_compiler_wraps_nonfinite_duplicate_term_aggregation(kind):
    if kind == "stoichiometry":
        aggregate = FluxModel(
            (FluxMetabolite("A", True),),
            (
                FluxReaction(
                    "R",
                    (
                        StoichiometricTerm("A", 1e308),
                        StoichiometricTerm("A", 1e308),
                    ),
                    0.0,
                    1.0,
                ),
            ),
            LinearObjective("maximise", (ObjectiveTerm("R", 1.0),)),
        )
        message = "aggregated stoichiometric.*non-finite.*reaction 'R'.*metabolite 'A'"
    else:
        aggregate = FluxModel(
            (),
            (FluxReaction("R", (), 0.0, 1.0),),
            LinearObjective(
                "maximise",
                (ObjectiveTerm("R", 1e308), ObjectiveTerm("R", 1e308)),
            ),
        )
        message = "aggregated objective coefficient.*non-finite.*reaction 'R'"

    with pytest.raises(AnalysisError, match=message):
        compile_flux_lp(aggregate)


def test_primal_validator_fails_closed_on_nonfinite_mass_arithmetic():
    overflow = FluxModel(
        (FluxMetabolite("A", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("A", 1e308),), 0.0, 2.0
            ),
            FluxReaction(
                "Y", (StoichiometricTerm("A", -1e308),), 0.0, 2.0
            ),
            FluxReaction("Q", (), 1.0, 1.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("Q", 1.0),)),
    )
    lp = compile_flux_lp(overflow)

    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(AnalysisError, match="non-finite.*mass|mass.*non-finite"):
            highs_module._validate(lp, (2.0, 2.0, 1.0), 1.0, (0.0, 0.0, 1.0))


def test_primal_validator_wraps_nonfinite_objective_arithmetic():
    overflow = FluxModel(
        (),
        (
            FluxReaction("X", (), 0.0, 2.0),
            FluxReaction("Y", (), 0.0, 2.0),
            FluxReaction("Q", (), 1.0, 1.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("Q", 1.0),)),
    )
    lp = compile_flux_lp(overflow)

    with pytest.raises(AnalysisError, match="non-finite.*objective|objective.*non-finite"):
        highs_module._validate(
            lp,
            (2.0, 2.0, 1.0),
            0.0,
            (1e308, -1e308, 0.0),
        )


def test_native_import_firewall_and_lazy_highspy_import():
    source_modules = set(sys.modules)
    assert "cobra" not in __import__("fluxemu.flux_analysis.highs", fromlist=["x"]).__dict__
    source = __import__("inspect").getsource(__import__("fluxemu.flux_analysis.highs", fromlist=["x"]))
    assert all(token not in source for token in ("import cobra", "import optlang", "import mfapy", "scipy.optimize"))
    assert "highspy" not in source_modules or importlib.util.find_spec("highspy") is not None
