"""Focused unit tests for native affine-hull complete-state sampling."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math

import numpy as np
import pandas as pd
import pytest

import fluxemu.flux_analysis.highs as highs
import fluxemu.flux_analysis.sampling as sampling
from fluxemu.exceptions import AnalysisError
from fluxemu.execution import CanonicalFluxState
from fluxemu.flux_analysis import (
    FBAResult,
    FVAResult,
    PreparedFluxRegion,
    RetainedObjectiveConstraint,
    prepare_highs_flux_region,
    run_prepared_highs_vffva,
    sample_highs_flux_states,
    sample_prepared_highs_flux_states,
    validate_flux_states,
)
from fluxemu.model import (
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    LinearObjective,
    ObjectiveTerm,
    StoichiometricTerm,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("highspy") is None, reason="highspy is unavailable"
)


def _split_model() -> FluxModel:
    return FluxModel(
        (FluxMetabolite("internal", True),),
        (
            FluxReaction(
                "SOURCE", (StoichiometricTerm("internal", 1.0),), 0.0, 10.0
            ),
            FluxReaction(
                "PATH_B", (StoichiometricTerm("internal", -1.0),), 0.0, 10.0
            ),
            FluxReaction(
                "PATH_A", (StoichiometricTerm("internal", -1.0),), 0.0, 10.0
            ),
            FluxReaction("FIXED", (), 2.0, 2.0),
            FluxReaction("SIGNED", (), -3.0, 4.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("SOURCE", 1.0),)),
    )


def _unique_model() -> FluxModel:
    return FluxModel(
        (FluxMetabolite("A", True),),
        (
            FluxReaction("IN", (StoichiometricTerm("A", 1.0),), 0.0, 5.0),
            FluxReaction("OUT", (StoichiometricTerm("A", -1.0),), 0.0, 5.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("OUT", 1.0),)),
    )


def _overflow_lp(*, balanced: bool) -> highs.CompiledFluxLP:
    lp = highs.CompiledFluxLP(
        reaction_ids=("X", "Y"),
        balanced_metabolite_ids=("A",) if balanced else (),
        row_starts=(0, 2) if balanced else (0,),
        column_indices=(0, 1) if balanced else (),
        coefficients=(1e308, -1e308) if balanced else (),
        lower_bounds=(-2.0, -2.0),
        upper_bounds=(2.0, 2.0),
        objective_coefficients=(1e308, -1e308),
        objective_direction="max",
        fingerprint="",
    )
    return replace(lp, fingerprint=highs._compiled_lp_fingerprint(lp))


def test_affine_hit_and_run_replays_without_touching_global_numpy_rng() -> None:
    model = _split_model()
    arguments = dict(
        count=8,
        fraction_of_optimum=0.8,
        seed=1203,
        burn_in=12,
        thinning=2,
        workers=1,
    )
    np.random.seed(884)
    expected_global = np.random.random(4)
    np.random.seed(884)
    first = sample_highs_flux_states(model, **arguments)
    observed_global = np.random.random(4)
    second = sample_highs_flux_states(model, **arguments)

    assert np.array_equal(observed_global, expected_global)
    assert first == second
    assert first.validation.valid
    assert first.provenance.algorithm == "affine_hit_and_run"
    assert first.provenance.rng == "numpy.PCG64"
    assert first.provenance.total_steps == 28
    assert first.provenance.affine_dimension == 3
    assert first.provenance.numerically_fixed_reactions == ("FIXED",)
    assert not first.provenance.objective_face
    assert len({state.values for state in first.states}) > 1
    for state in first.states:
        values = dict(state.values)
        assert tuple(identifier for identifier, _ in state.values) == (
            "SOURCE",
            "PATH_B",
            "PATH_A",
            "FIXED",
            "SIGNED",
        )
        assert values["SOURCE"] == pytest.approx(
            values["PATH_B"] + values["PATH_A"], abs=1e-7
        )
        assert values["SOURCE"] >= 8.0 - 1e-7


def test_fraction_one_adds_objective_equality_but_retains_face_dimensions() -> None:
    result = sample_highs_flux_states(
        _split_model(),
        7,
        1.0,
        seed=55,
        burn_in=10,
        thinning=2,
        workers=1,
    )

    assert result.provenance.objective_face
    assert result.provenance.affine_dimension == 2
    assert result.validation.optimal_face_valid
    assert all(dict(state.values)["SOURCE"] == pytest.approx(10.0) for state in result.states)
    assert len({dict(state.values)["PATH_A"] for state in result.states}) > 1


def test_small_scaled_fractional_objective_is_not_collapsed_to_a_face() -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("SMALL", (), 0.0, 1e-8),
            FluxReaction("FREE", (), -1.0, 1.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("SMALL", 1.0),)),
    )
    result = sample_highs_flux_states(
        model,
        8,
        0.5,
        seed=9,
        burn_in=8,
        thinning=1,
        workers=1,
    )

    assert not result.provenance.objective_face
    assert result.provenance.affine_dimension == 2
    small_values = [dict(state.values)["SMALL"] for state in result.states]
    assert min(small_values) >= 5e-9 - 1e-15
    assert len(set(small_values)) > 1


def test_full_svd_nullspace_retains_all_underdetermined_dimensions() -> None:
    prepared = prepare_highs_flux_region(_split_model(), 0.8)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    minima, maxima = sampling._validate_fva_result(prepared, fva)
    geometry = sampling._build_geometry(prepared, minima, maxima)

    # One mass-balance equality plus one fixed coordinate among five fluxes.
    assert geometry.basis.shape == (5, 3)
    stoichiometry = sampling._dense_stoichiometry(prepared)
    assert np.max(np.abs(stoichiometry @ geometry.basis)) < 1e-12
    assert np.max(np.abs(geometry.basis[3])) < 1e-12


def test_numerically_degenerate_chords_retry_then_fail_contextually(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_highs_flux_region(_split_model(), 0.8)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    calls = 0

    def degenerate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (0.0, sampling.DIRECTION_TOLERANCE / 2.0)

    monkeypatch.setattr(sampling, "_line_chord", degenerate)
    with pytest.raises(
        AnalysisError,
        match="step 1 after 4 direction attempts.*numerically degenerate",
    ):
        sample_prepared_highs_flux_states(
            prepared,
            fva,
            1,
            seed=1,
            burn_in=0,
            thinning=1,
            max_direction_attempts=4,
        )
    assert calls == 4


def test_unbounded_line_direction_is_rejected_without_truncation() -> None:
    prepared = prepare_highs_flux_region(_split_model(), 0.8)
    signed_index = prepared.lp.reaction_ids.index("SIGNED")
    upper_bounds = list(prepared.lp.upper_bounds)
    upper_bounds[signed_index] = math.inf
    forged = replace(
        prepared,
        lp=replace(prepared.lp, upper_bounds=tuple(upper_bounds)),
    )
    point = np.asarray(tuple(float(value) for value in prepared.fba.fluxes))
    direction = np.zeros(len(point))
    direction[signed_index] = 1.0

    with pytest.raises(AnalysisError, match="unbounded.*arbitrary truncation"):
        sampling._line_chord(
            forged, point, direction, objective_face=False
        )


def test_zero_dimensional_region_repeats_values_with_unique_sample_ids() -> None:
    result = sample_highs_flux_states(
        _unique_model(), 4, 1.0, seed=5, burn_in=50, thinning=7, workers=1
    )

    assert result.provenance.affine_dimension == 0
    assert result.provenance.total_steps == 0
    assert len({state.sample_id for state in result.states}) == 4
    assert {state.values for state in result.states} == {
        (("IN", 5.0), ("OUT", 5.0))
    }


def test_prepared_sampler_does_not_compile_or_resolve_the_objective_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_highs_flux_region(_split_model(), 0.8)
    fva = run_prepared_highs_vffva(prepared, workers=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared sampling repeated model preparation")

    monkeypatch.setattr(highs, "compile_flux_lp", forbidden)
    monkeypatch.setattr(highs, "_run_compiled_fba", forbidden)
    result = sample_prepared_highs_flux_states(
        prepared, fva, 2, seed=7, burn_in=2, thinning=1
    )
    assert result.validation.valid


def test_mutable_fva_forgery_is_rejected_before_geometry() -> None:
    prepared = prepare_highs_flux_region(_split_model(), 0.8)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    forged_ranges = fva.ranges.copy()
    forged_ranges.loc["SOURCE", "minimum"] = 9.0
    forged_ranges.loc["SOURCE", "maximum"] = 9.5
    forged = replace(fva, ranges=forged_ranges)

    with pytest.raises(AnalysisError, match="does not bracket.*FBA primal"):
        sample_prepared_highs_flux_states(
            prepared, forged, 1, burn_in=0, thinning=1
        )


def test_provenance_fingerprints_the_exact_fva_geometry() -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("OBJECTIVE", (), 1.0, 1.0),
            FluxReaction("X", (), 0.0, 10.0),
            FluxReaction("Y", (), 0.0, 10.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("OBJECTIVE", 1.0),)),
    )
    prepared = prepare_highs_flux_region(model)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    altered_ranges = fva.ranges.copy()
    altered_ranges.loc["X", "maximum"] = 8.0
    altered = replace(fva, ranges=altered_ranges)

    arguments = dict(count=3, seed=11, burn_in=3, thinning=1)
    original_result = sample_prepared_highs_flux_states(
        prepared, fva, **arguments
    )
    altered_result = sample_prepared_highs_flux_states(
        prepared, altered, **arguments
    )

    original_fingerprint = original_result.provenance.fva_ranges_fingerprint
    altered_fingerprint = altered_result.provenance.fva_ranges_fingerprint
    assert len(original_fingerprint) == len(altered_fingerprint) == 64
    assert original_fingerprint != altered_fingerprint
    assert original_result.provenance != altered_result.provenance
    assert original_result.states != altered_result.states


def test_nonrepresentable_fva_width_fails_contextually_without_numpy_warning() -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("OBJECTIVE", (), 1.0, 1.0),
            FluxReaction("WIDE", (), -1e308, 1e308),
        ),
        LinearObjective("maximise", (ObjectiveTerm("OBJECTIVE", 1.0),)),
    )
    lp = highs.compile_flux_lp(model)
    fba = FBAResult(
        1.0,
        "optimal",
        "max",
        pd.Series((1.0, 0.0), index=lp.reaction_ids, dtype=float),
    )
    prepared = PreparedFluxRegion(
        lp=lp,
        fba=fba,
        retention=RetainedObjectiveConstraint(1.0, 1.0, 1.0, ">=", "max"),
        flux_model=model,
        flux_model_fingerprint=highs._flux_model_fingerprint(model),
    )
    fva = FVAResult(
        pd.DataFrame(
            {"minimum": (1.0, -1e308), "maximum": (1.0, 1e308)},
            index=lp.reaction_ids,
        ),
        1.0,
        1.0,
        "max",
    )

    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(
            AnalysisError,
            match="FVA range width for reaction 'WIDE'.*numerically unrepresentable",
        ):
            sample_prepared_highs_flux_states(
                prepared, fva, 1, seed=1, burn_in=0, thinning=1
            )


@pytest.mark.parametrize(
    "case",
    (
        "row-count",
        "row-origin",
        "column-index",
        "coefficient-count",
    ),
)
def test_prepared_sampler_rejects_malformed_csr_even_with_recomputed_fingerprint(
    case: str,
) -> None:
    prepared = prepare_highs_flux_region(_split_model(), 0.8)
    changes: dict[str, tuple[object, ...]]
    if case == "row-count":
        changes = {"row_starts": (0,)}
    elif case == "row-origin":
        changes = {"row_starts": (1, len(prepared.lp.column_indices))}
    elif case == "column-index":
        changes = {
            "column_indices": (len(prepared.lp.reaction_ids),)
            + prepared.lp.column_indices[1:]
        }
    else:
        changes = {"coefficients": prepared.lp.coefficients[:-1]}
    malformed_lp = replace(prepared.lp, **changes)
    malformed_lp = replace(
        malformed_lp,
        fingerprint=highs._compiled_lp_fingerprint(malformed_lp),
    )
    forged = replace(prepared, lp=malformed_lp)

    with pytest.raises(AnalysisError, match="prepared compiled LP"):
        highs._validate_prepared_flux_region(forged)


def test_validator_reports_face_error_and_duplicate_sample_ids() -> None:
    model = _split_model()
    values = (
        ("SOURCE", 9.0),
        ("PATH_B", 4.0),
        ("PATH_A", 5.0),
        ("FIXED", 2.0),
        ("SIGNED", 0.0),
    )
    states = (
        CanonicalFluxState("duplicate", values),
        CanonicalFluxState("duplicate", values),
    )
    report = validate_flux_states(
        model,
        states,
        retained_objective_bound=8.0,
        objective_direction="max",
        optimal_face_value=10.0,
    )

    assert not report.valid
    assert not report.unique_sample_ids_valid
    assert not report.optimal_face_valid
    assert report.max_optimal_face_error == pytest.approx(1.0)
    assert all(detail.optimal_face_error == pytest.approx(1.0) for detail in report.details)


def test_validator_rejects_nonfinite_derived_objective_from_finite_inputs() -> None:
    model = FluxModel(
        (),
        (FluxReaction("X", (), 2.0, 2.0),),
        LinearObjective("maximise", (ObjectiveTerm("X", 1e308),)),
    )
    state = CanonicalFluxState("overflow", (("X", 2.0),))

    report = validate_flux_states(
        model,
        (state,),
        retained_objective_bound=0.0,
        objective_direction="max",
        optimal_face_value=0.0,
    )

    assert not report.valid
    assert report.finite_values_valid
    assert not report.retained_objective_valid
    assert not report.optimal_face_valid
    assert math.isinf(report.max_retained_objective_violation)
    assert math.isinf(report.max_optimal_face_error)
    assert math.isinf(report.details[0].objective_value)
    assert "derived biological objective is non-finite" in report.details[0].errors


def test_validator_rejects_nonfinite_derived_mass_balance_residual() -> None:
    model = FluxModel(
        (FluxMetabolite("A", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("A", 1e308),), 2.0, 2.0
            ),
        ),
        LinearObjective("maximise", (ObjectiveTerm("X", 1.0),)),
    )
    state = CanonicalFluxState("overflow", (("X", 2.0),))

    report = validate_flux_states(
        model,
        (state,),
        retained_objective_bound=0.0,
        objective_direction="max",
    )

    assert not report.valid
    assert report.finite_values_valid
    assert not report.mass_balance_valid
    assert math.isinf(report.max_mass_balance_residual)
    assert report.details[0].max_mass_balance_metabolite == "A"
    assert any(
        "derived mass-balance residual is non-finite" in error
        for error in report.details[0].errors
    )


def test_validator_uses_stable_mass_balance_summation() -> None:
    model = FluxModel(
        (FluxMetabolite("A", True),),
        (
            FluxReaction("BIGP", (StoichiometricTerm("A", 1e16),), 1.0, 1.0),
            FluxReaction("ONE", (StoichiometricTerm("A", 1.0),), 1.0, 1.0),
            FluxReaction("BIGN", (StoichiometricTerm("A", -1e16),), 1.0, 1.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("ONE", 1.0),)),
    )
    state = CanonicalFluxState(
        "cancellation",
        (("BIGP", 1.0), ("ONE", 1.0), ("BIGN", 1.0)),
    )

    report = validate_flux_states(
        model,
        (state,),
        retained_objective_bound=1.0,
        objective_direction="max",
        optimal_face_value=1.0,
    )

    assert not report.valid
    assert not report.mass_balance_valid
    assert report.max_mass_balance_residual == pytest.approx(1.0)
    assert report.details[0].max_mass_balance_metabolite == "A"


def test_duplicate_terms_use_stable_aggregation_in_compiler_and_validator() -> None:
    model = FluxModel(
        (FluxMetabolite("A", True),),
        (
            FluxReaction(
                "R",
                (
                    StoichiometricTerm("A", 1e16),
                    StoichiometricTerm("A", 1.0),
                    StoichiometricTerm("A", -1e16),
                ),
                1.0,
                1.0,
            ),
        ),
        LinearObjective(
            "maximise",
            (
                ObjectiveTerm("R", 1e16),
                ObjectiveTerm("R", 1.0),
                ObjectiveTerm("R", -1e16),
            ),
        ),
    )

    lp = highs.compile_flux_lp(model)
    assert lp.coefficients == (1.0,)
    assert lp.objective_coefficients == (1.0,)
    report = validate_flux_states(
        model,
        (CanonicalFluxState("stable", (("R", 1.0),)),),
        retained_objective_bound=1.0,
        objective_direction="max",
        optimal_face_value=1.0,
    )
    assert not report.valid
    assert report.max_mass_balance_residual == pytest.approx(1.0)
    assert report.details[0].objective_value == pytest.approx(1.0)


def test_validator_rejects_nonfinite_aggregated_objective_at_zero_flux() -> None:
    model = FluxModel(
        (),
        (FluxReaction("X", (), 0.0, 0.0),),
        LinearObjective(
            "maximise",
            (ObjectiveTerm("X", 1e308), ObjectiveTerm("X", 1e308)),
        ),
    )

    with pytest.raises(
        AnalysisError,
        match="aggregated objective coefficient.*non-finite.*reaction 'X'",
    ):
        validate_flux_states(
            model,
            (CanonicalFluxState("zero", (("X", 0.0),)),),
            retained_objective_bound=0.0,
            objective_direction="max",
        )


def test_highs_primal_validator_rejects_nonfinite_mass_balance_arithmetic() -> None:
    lp = _overflow_lp(balanced=True)

    with pytest.raises(
        AnalysisError,
        match="mass balance for metabolite 'A'.*non-finite product",
    ):
        highs._validate(lp, (2.0, 2.0), 0.0, (0.0, 0.0))


def test_highs_primal_validator_rejects_nonfinite_objective_arithmetic() -> None:
    lp = _overflow_lp(balanced=False)

    with pytest.raises(
        AnalysisError,
        match="objective recalculation.*non-finite product",
    ):
        highs._validate(lp, (2.0, 2.0), 0.0, (1e308, -1e308))


def test_cold_fva_rejects_nonfinite_retained_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("X", (), 2.0, 2.0),
            FluxReaction("Y", (), 2.0, 2.0),
        ),
        LinearObjective(
            "maximise",
            (ObjectiveTerm("X", 1e308), ObjectiveTerm("Y", -1e308)),
        ),
    )

    monkeypatch.setattr(
        highs,
        "_run_compiled_fba",
        lambda lp: FBAResult(
            0.0,
            "optimal",
            "max",
            pd.Series((2.0, 2.0), index=lp.reaction_ids, dtype=float),
        ),
    )
    monkeypatch.setattr(
        highs,
        "_solve",
        lambda lp, costs, direction, operation, retention=None: (
            [2.0, 2.0],
            2.0,
            "Optimal",
        ),
    )
    monkeypatch.setattr(highs, "_validate", lambda *args, **kwargs: None)

    with pytest.raises(
        AnalysisError,
        match="FVA endpoint for reaction 'X' retained objective.*non-finite product",
    ):
        highs.run_highs_fva_reference(model)


def test_reusable_endpoint_rejects_nonfinite_retained_objective() -> None:
    lp = _overflow_lp(balanced=False)

    class _Solution:
        col_value = (2.0, 2.0)

    class _Solver:
        def changeColCost(self, *args):
            return None

        def setMinimize(self):
            return None

        def run(self):
            return None

        def getModelStatus(self):
            return highs._highspy().HighsModelStatus.kOptimal

        def modelStatusToString(self, status):
            return "Optimal"

        def getSolution(self):
            return _Solution()

        def getObjectiveValue(self):
            return 2.0

    worker = object.__new__(highs._ReusableFVAWorker)
    worker.lp = lp
    worker.retention = (">=", 0.0)
    worker.endpoint_count = 0
    worker.solver = _Solver()

    with pytest.raises(
        AnalysisError,
        match="FVA minimum for reaction 'X'.*retained biological objective.*non-finite product",
    ):
        worker.solve((0, "min"))


def test_line_chord_uses_stable_retained_objective_summation() -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("P", (), -10.0, 10.0),
            FluxReaction("ONE", (), -10.0, 10.0),
            FluxReaction("N", (), -10.0, 10.0),
        ),
        LinearObjective(
            "maximise",
            (
                ObjectiveTerm("P", 1e16),
                ObjectiveTerm("ONE", 1.0),
                ObjectiveTerm("N", -1e16),
            ),
        ),
    )
    lp = highs.compile_flux_lp(model)
    prepared = PreparedFluxRegion(
        lp=lp,
        fba=FBAResult(
            1.0,
            "optimal",
            "max",
            pd.Series((1.0, 1.0, 1.0), index=lp.reaction_ids, dtype=float),
        ),
        retention=RetainedObjectiveConstraint(0.5, 1.0, 0.5, ">=", "max"),
        flux_model=model,
        flux_model_fingerprint=highs._flux_model_fingerprint(model),
    )

    lower, upper = sampling._line_chord(
        prepared,
        np.asarray((1.0, 1.0, 1.0)),
        np.asarray((0.0, 1.0, 0.0)),
        objective_face=False,
    )

    assert lower == pytest.approx(-0.5)
    assert upper == pytest.approx(9.0)


def test_compiler_rejects_nonfinite_aggregated_stoichiometry() -> None:
    model = FluxModel(
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

    with pytest.raises(
        AnalysisError,
        match="aggregated stoichiometric.*non-finite.*reaction 'R'.*metabolite 'A'",
    ):
        highs.compile_flux_lp(model)


def test_compiler_rejects_nonfinite_aggregated_objective() -> None:
    model = FluxModel(
        (),
        (FluxReaction("R", (), 0.0, 1.0),),
        LinearObjective(
            "maximise",
            (ObjectiveTerm("R", 1e308), ObjectiveTerm("R", 1e308)),
        ),
    )

    with pytest.raises(
        AnalysisError,
        match="aggregated objective coefficient.*non-finite.*reaction 'R'",
    ):
        highs.compile_flux_lp(model)


def test_sampler_never_mutates_the_supplied_fva_frame() -> None:
    prepared = prepare_highs_flux_region(_split_model(), 0.8)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    expected = fva.ranges.copy(deep=True)

    sample_prepared_highs_flux_states(
        prepared, fva, 2, seed=8, burn_in=2, thinning=1
    )

    pd.testing.assert_frame_equal(fva.ranges, expected)
