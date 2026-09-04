"""Independent black-box acceptance gates for native feasible-state sampling."""

from __future__ import annotations

from dataclasses import replace
import math
import re
import subprocess
import sys

import numpy as np
import pytest

from fluxemu.exceptions import AnalysisError
from fluxemu.execution import CanonicalFluxState
from fluxemu.flux_analysis import (
    prepare_highs_flux_region,
    run_highs_vffva,
    sample_highs_flux_states,
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
from fluxemu.real_model import load_ecoli_core_flux_model


TOLERANCE = 1e-7


def _joint_model() -> FluxModel:
    """Return a bounded face with correlated paths and fixed/signed variables."""

    return FluxModel(
        (
            FluxMetabolite("source", False),
            FluxMetabolite("internal", True),
            FluxMetabolite("product", False),
        ),
        (
            # Deliberately non-lexical order is part of the scientific contract.
            FluxReaction(
                "Z_SOURCE",
                (
                    StoichiometricTerm("source", -1.0),
                    StoichiometricTerm("internal", 1.0),
                ),
                0.0,
                10.0,
            ),
            FluxReaction(
                "PATH_B",
                (
                    StoichiometricTerm("internal", -1.0),
                    StoichiometricTerm("product", 1.0),
                ),
                0.0,
                10.0,
            ),
            FluxReaction(
                "PATH_A",
                (
                    StoichiometricTerm("internal", -1.0),
                    StoichiometricTerm("product", 1.0),
                ),
                0.0,
                10.0,
            ),
            FluxReaction("SIGNED", (), -3.0, 4.0),
            FluxReaction("FIXED", (), 2.0, 2.0),
            FluxReaction("BLOCKED", (), 0.0, 0.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("Z_SOURCE", 1.0),)),
    )


def _minimum_model() -> FluxModel:
    return FluxModel(
        (FluxMetabolite("external", False),),
        (
            FluxReaction(
                "COST",
                (StoichiometricTerm("external", 1.0),),
                -10.0,
                -2.0,
            ),
            FluxReaction("AUXILIARY", (), -3.0, 3.0),
            FluxReaction("FIXED", (), 2.0, 2.0),
            FluxReaction("BLOCKED", (), 0.0, 0.0),
        ),
        LinearObjective("minimise", (ObjectiveTerm("COST", 1.0),)),
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


def _values(state: CanonicalFluxState) -> dict[str, float]:
    return dict(state.values)


def _state(values: tuple[tuple[str, float], ...], sample_id: str = "manual"):
    return CanonicalFluxState(sample_id, values)


def _valid_manual_values() -> tuple[tuple[str, float], ...]:
    return (
        ("Z_SOURCE", 9.0),
        ("PATH_B", 4.0),
        ("PATH_A", 5.0),
        ("SIGNED", 0.0),
        ("FIXED", 2.0),
        ("BLOCKED", 0.0),
    )


def _validate_manual(state: CanonicalFluxState):
    return validate_flux_states(
        _joint_model(),
        (state,),
        retained_objective_bound=8.0,
        objective_direction="max",
    )


def test_fractional_maximum_replays_complete_correlated_ordered_states() -> None:
    model = _joint_model()
    arguments = dict(
        count=12,
        fraction_of_optimum=0.8,
        seed=1729,
        burn_in=20,
        thinning=3,
        workers=1,
        max_direction_attempts=100,
    )
    first = sample_highs_flux_states(model, **arguments)
    second = sample_highs_flux_states(model, **arguments)

    assert first.states == second.states
    assert first.provenance == second.provenance
    assert first.sample_count == len(first.states) == 12
    assert first.validation.valid
    assert first.validation.sample_count == 12
    assert len({state.sample_id for state in first.states}) == 12

    reaction_order = tuple(reaction.reaction_id for reaction in model.reactions)
    prepared = prepare_highs_flux_region(model, 0.8)
    provenance = first.provenance
    assert provenance.algorithm == first.algorithm == "affine_hit_and_run"
    assert provenance.algorithm_version == "1"
    assert provenance.seed == first.seed == 1729
    assert provenance.rng == "numpy.PCG64"
    assert provenance.fraction_of_optimum == first.fraction_of_optimum == 0.8
    assert (
        provenance.biological_optimum
        == first.biological_optimum
        == pytest.approx(10.0)
    )
    assert (
        provenance.retained_objective_bound
        == first.retained_objective_bound
        == pytest.approx(8.0)
    )
    assert provenance.retained_objective_sense == ">="
    assert provenance.objective_direction == first.objective_direction == "max"
    assert provenance.compiled_lp_fingerprint == prepared.lp.fingerprint
    assert provenance.flux_model_fingerprint == prepared.flux_model_fingerprint
    assert provenance.fva_ranges_fingerprint == (
        second.provenance.fva_ranges_fingerprint
    )
    assert re.fullmatch(r"[0-9a-f]{64}", provenance.fva_ranges_fingerprint)
    assert provenance.model_fingerprint == first.model_fingerprint
    assert provenance.reaction_ids == first.reaction_ids == reaction_order
    assert provenance.burn_in == 20
    assert provenance.thinning == 3
    assert provenance.total_steps == 20 + 12 * 3
    assert 0 <= provenance.direction_retries < (
        provenance.total_steps * provenance.max_direction_attempts
    )
    assert provenance.max_direction_attempts == 100
    assert provenance.affine_dimension == 3
    assert provenance.numerically_fixed_reactions == ("FIXED", "BLOCKED")
    assert provenance.objective_face is False
    assert provenance.fixed_range_tolerance == pytest.approx(1e-12)
    assert provenance.rank_tolerance == pytest.approx(1e-12)
    assert provenance.direction_tolerance == pytest.approx(1e-12)
    assert provenance.bounds_tolerance == pytest.approx(TOLERANCE)
    assert provenance.mass_balance_tolerance == pytest.approx(TOLERANCE)
    assert provenance.objective_tolerance == pytest.approx(TOLERANCE)
    assert math.isfinite(provenance.center_slack)
    assert provenance.center_slack > provenance.direction_tolerance
    assert provenance.numpy_version == np.__version__
    assert provenance.highs_version

    validation = first.validation
    assert validation.unique_sample_ids_valid
    assert validation.reaction_membership_valid
    assert validation.reaction_order_valid
    assert validation.finite_values_valid
    assert validation.bounds_valid
    assert validation.mass_balance_valid
    assert validation.retained_objective_valid
    assert validation.optimal_face_valid
    assert validation.retained_objective_bound == pytest.approx(
        provenance.retained_objective_bound
    )
    assert validation.objective_direction == provenance.objective_direction
    assert validation.optimal_face_value is None
    assert validation.bounds_tolerance == provenance.bounds_tolerance
    assert validation.mass_balance_tolerance == provenance.mass_balance_tolerance
    assert validation.objective_tolerance == provenance.objective_tolerance
    assert tuple(detail.sample_id for detail in validation.details) == tuple(
        state.sample_id for state in first.states
    )
    assert all(detail.valid for detail in validation.details)
    assert all(detail.reaction_membership_valid for detail in validation.details)
    assert all(detail.reaction_order_valid for detail in validation.details)
    assert all(detail.finite_values_valid for detail in validation.details)
    assert all(detail.bounds_valid for detail in validation.details)
    assert all(detail.mass_balance_valid for detail in validation.details)
    assert all(detail.retained_objective_valid for detail in validation.details)
    assert all(detail.optimal_face_valid for detail in validation.details)

    observed = []
    for state in first.states:
        assert tuple(reaction_id for reaction_id, _ in state.values) == reaction_order
        flux = _values(state)
        assert all(math.isfinite(value) for value in flux.values())
        assert flux["Z_SOURCE"] == pytest.approx(
            flux["PATH_A"] + flux["PATH_B"], abs=TOLERANCE
        )
        assert flux["Z_SOURCE"] >= 8.0 - TOLERANCE
        assert -3.0 - TOLERANCE <= flux["SIGNED"] <= 4.0 + TOLERANCE
        assert flux["FIXED"] == pytest.approx(2.0, abs=TOLERANCE)
        assert flux["BLOCKED"] == pytest.approx(0.0, abs=TOLERANCE)
        observed.append(tuple(value for _, value in state.values))
    assert len(set(observed)) > 1

    # Reaction-wise minima are valid extrema but not a jointly feasible state.
    fva = run_highs_vffva(model, 0.8, workers=1)
    minimum_endpoints = _state(
        tuple(
            (reaction_id, float(fva.ranges.loc[reaction_id, "minimum"]))
            for reaction_id in reaction_order
        ),
        "fva-minima",
    )
    endpoint_report = validate_flux_states(
        model,
        (minimum_endpoints,),
        retained_objective_bound=8.0,
        objective_direction="max",
    )
    assert not endpoint_report.valid
    assert not endpoint_report.mass_balance_valid
    assert endpoint_report.max_mass_balance_residual == pytest.approx(8.0)


def test_fraction_one_samples_the_maximum_optimal_face_deliberately() -> None:
    result = sample_highs_flux_states(
        _joint_model(),
        8,
        1.0,
        seed=2718,
        burn_in=16,
        thinning=2,
        workers=1,
    )

    assert result.validation.valid
    assert result.validation.optimal_face_valid
    for state in result.states:
        flux = _values(state)
        assert flux["Z_SOURCE"] == pytest.approx(10.0, abs=TOLERANCE)
        assert flux["PATH_A"] + flux["PATH_B"] == pytest.approx(
            10.0, abs=TOLERANCE
        )
    assert len({tuple(state.values) for state in result.states}) > 1


def test_fraction_below_one_with_zero_optimum_remains_on_implicit_face() -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("ZERO_OBJECTIVE", (), -5.0, 0.0),
            FluxReaction("FREE", (), -2.0, 3.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("ZERO_OBJECTIVE", 1.0),)),
    )
    result = sample_highs_flux_states(
        model,
        6,
        0.5,
        seed=1618,
        burn_in=10,
        thinning=2,
        workers=1,
    )

    assert result.validation.valid
    assert all(
        _values(state)["ZERO_OBJECTIVE"] == pytest.approx(0.0, abs=TOLERANCE)
        for state in result.states
    )
    assert len({_values(state)["FREE"] for state in result.states}) > 1


@pytest.mark.parametrize(
    ("fraction", "retained_bound"),
    ((1.0, -10.0), (0.9, -9.0)),
    ids=("optimal-face", "fractional"),
)
def test_minimizing_objective_uses_the_upper_retained_bound(
    fraction: float, retained_bound: float
) -> None:
    result = sample_highs_flux_states(
        _minimum_model(),
        7,
        fraction,
        seed=31415,
        burn_in=12,
        thinning=2,
        workers=1,
    )

    assert result.validation.valid
    for state in result.states:
        flux = _values(state)
        assert -10.0 - TOLERANCE <= flux["COST"] <= retained_bound + TOLERANCE
        assert -3.0 - TOLERANCE <= flux["AUXILIARY"] <= 3.0 + TOLERANCE
        assert flux["FIXED"] == pytest.approx(2.0, abs=TOLERANCE)
        assert flux["BLOCKED"] == pytest.approx(0.0, abs=TOLERANCE)
    if fraction == 1.0:
        assert result.validation.optimal_face_valid


def test_zero_dimensional_region_returns_the_unique_state_with_unique_ids() -> None:
    result = sample_highs_flux_states(
        _unique_model(),
        5,
        1.0,
        seed=42,
        burn_in=5,
        thinning=2,
        workers=1,
    )

    assert result.sample_count == 5
    assert result.validation.valid
    assert len({state.sample_id for state in result.states}) == 5
    assert {
        tuple(value for _, value in state.values) for state in result.states
    } == {(5.0, 5.0)}


def test_validator_rejects_reordered_missing_duplicate_and_unknown_reactions() -> None:
    values = _valid_manual_values()
    valid = _validate_manual(_state(values))
    assert valid.valid

    reordered = _validate_manual(_state((values[1], values[0]) + values[2:]))
    assert not reordered.valid
    assert reordered.reaction_membership_valid
    assert not reordered.reaction_order_valid
    assert not reordered.details[0].reaction_order_valid

    missing = _validate_manual(_state(values[:-1]))
    assert not missing.valid
    assert not missing.reaction_membership_valid
    assert missing.details[0].missing_reactions == ("BLOCKED",)

    duplicate_values = values[:2] + (("PATH_B", 5.0),) + values[3:]
    duplicate = _validate_manual(_state(duplicate_values))
    assert not duplicate.valid
    assert duplicate.details[0].duplicate_reactions == ("PATH_B",)
    assert duplicate.details[0].missing_reactions == ("PATH_A",)

    unknown_values = values[:-1] + (("UNKNOWN", 0.0),)
    unknown = _validate_manual(_state(unknown_values))
    assert not unknown.valid
    assert unknown.details[0].missing_reactions == ("BLOCKED",)
    assert unknown.details[0].unexpected_reactions == ("UNKNOWN",)

    malformed_values = values[:-1] + (("BLOCKED",),)
    malformed = _validate_manual(
        CanonicalFluxState("malformed", malformed_values)  # type: ignore[arg-type]
    )
    assert not malformed.valid
    assert not malformed.reaction_membership_valid
    assert malformed.details[0].errors


def test_validator_reports_nonfinite_bound_balance_and_objective_diagnostics() -> None:
    values = _valid_manual_values()

    nonfinite_values = values[:3] + (("SIGNED", math.nan),) + values[4:]
    nonfinite = _validate_manual(_state(nonfinite_values, "nonfinite"))
    assert not nonfinite.valid
    assert not nonfinite.finite_values_valid
    assert nonfinite.details[0].nonfinite_reactions == ("SIGNED",)

    lower_values = values[:3] + (("SIGNED", -4.5),) + values[4:]
    lower = _validate_manual(_state(lower_values, "lower"))
    assert not lower.bounds_valid
    assert lower.max_lower_bound_violation == pytest.approx(1.5)
    assert lower.details[0].max_lower_bound_reaction == "SIGNED"

    upper_values = values[:3] + (("SIGNED", 5.0),) + values[4:]
    upper = _validate_manual(_state(upper_values, "upper"))
    assert not upper.bounds_valid
    assert upper.max_upper_bound_violation == pytest.approx(1.0)
    assert upper.details[0].max_upper_bound_reaction == "SIGNED"

    imbalance_values = (
        ("Z_SOURCE", 9.0),
        ("PATH_B", 4.0),
        ("PATH_A", 4.0),
    ) + values[3:]
    imbalance = _validate_manual(_state(imbalance_values, "imbalance"))
    assert not imbalance.mass_balance_valid
    assert imbalance.max_mass_balance_residual == pytest.approx(1.0)
    assert imbalance.details[0].max_mass_balance_metabolite == "internal"

    objective_values = (
        ("Z_SOURCE", 7.0),
        ("PATH_B", 3.0),
        ("PATH_A", 4.0),
    ) + values[3:]
    objective = _validate_manual(_state(objective_values, "objective"))
    assert objective.mass_balance_valid
    assert not objective.retained_objective_valid
    assert objective.max_retained_objective_violation == pytest.approx(1.0)
    assert objective.details[0].objective_value == pytest.approx(7.0)
    assert objective.details[0].retained_objective_violation == pytest.approx(1.0)

    optimal_face = validate_flux_states(
        _joint_model(),
        (_state(_valid_manual_values(), "off-face"),),
        retained_objective_bound=10.0,
        objective_direction="max",
        optimal_face_value=10.0,
    )
    assert not optimal_face.valid
    assert not optimal_face.optimal_face_valid
    assert optimal_face.max_optimal_face_error == pytest.approx(1.0)
    assert optimal_face.details[0].optimal_face_error == pytest.approx(1.0)


@pytest.mark.parametrize(
    "overrides",
    (
        {"objective_direction": "min"},
        {"objective_direction": "sideways"},
        {"retained_objective_bound": math.nan},
        {"retained_objective_bound": math.inf},
        {"bounds_tolerance": math.nan},
        {"bounds_tolerance": -1.0},
        {"mass_balance_tolerance": math.inf},
        {"objective_tolerance": -1.0},
    ),
)
def test_validator_rejects_inconsistent_region_and_invalid_tolerances(
    overrides: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "retained_objective_bound": 8.0,
        "objective_direction": "max",
    }
    arguments.update(overrides)
    with pytest.raises(AnalysisError):
        validate_flux_states(
            _joint_model(),
            (_state(_valid_manual_values()),),
            **arguments,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "arguments",
    (
        {"count": 0},
        {"count": -1},
        {"count": True},
        {"count": 1.5},
        {"count": 1, "fraction_of_optimum": 0.0},
        {"count": 1, "fraction_of_optimum": 1.01},
        {"count": 1, "fraction_of_optimum": math.nan},
        {"count": 1, "fraction_of_optimum": True},
        {"count": 1, "seed": -1},
        {"count": 1, "seed": True},
        {"count": 1, "seed": 1.5},
        {"count": 1, "burn_in": -1},
        {"count": 1, "thinning": 0},
        {"count": 1, "max_direction_attempts": 0},
        {"count": 1, "workers": 0},
    ),
)
def test_invalid_sampler_arguments_fail_explicitly(arguments: dict[str, object]) -> None:
    with pytest.raises(AnalysisError):
        sample_highs_flux_states(_joint_model(), **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("lower", "upper"),
    ((-3.0, math.inf), (-math.inf, 4.0)),
)
def test_nonfinite_bounds_fail_as_possible_unbounded_sampling_regions(
    lower: float, upper: float
) -> None:
    model = _joint_model()
    signed = replace(model.reactions[3], lower_bound=lower, upper_bound=upper)
    unbounded = replace(
        model,
        reactions=model.reactions[:3] + (signed,) + model.reactions[4:],
    )
    with pytest.raises(AnalysisError, match="(?i)(unbounded|non.?finite)"):
        sample_highs_flux_states(unbounded, 1, workers=1)


@pytest.mark.parametrize(
    ("direction", "bounds", "optimum", "retained_bound"),
    (
        ("minimise", (2.0, 10.0), 2.0, 1.8),
        ("maximise", (-10.0, -2.0), -2.0, -1.8),
    ),
)
def test_fractional_sign_crossing_objective_fails_early_instead_of_stalling(
    direction: str,
    bounds: tuple[float, float],
    optimum: float,
    retained_bound: float,
) -> None:
    model = FluxModel(
        (),
        (FluxReaction("OBJECTIVE", (), *bounds),),
        LinearObjective(direction, (ObjectiveTerm("OBJECTIVE", 1.0),)),
    )
    # Multiplication by f crosses past the optimum for a positive minimum or
    # negative maximum. Reject that empty region before direction retries.
    with pytest.raises(AnalysisError) as captured:
        sample_highs_flux_states(
            model,
            1,
            0.9,
            seed=1,
            burn_in=1,
            thinning=1,
            workers=1,
            max_direction_attempts=1,
        )
    message = str(captured.value)
    expected_direction = "min" if direction == "minimise" else "max"
    assert f"direction={expected_direction}" in message
    assert f"biological_optimum={optimum:g}" in message
    assert "fraction_of_optimum=0.9" in message
    assert f"retained_bound={retained_bound:g}" in message


@pytest.mark.parametrize(
    ("direction", "bounds", "optimum"),
    (
        ("minimise", (2.0, 10.0), 2.0),
        ("maximise", (-10.0, -2.0), -2.0),
    ),
)
def test_sign_crossing_objectives_are_valid_at_fraction_one(
    direction: str, bounds: tuple[float, float], optimum: float
) -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("OBJECTIVE", (), *bounds),
            FluxReaction("FREE", (), -1.0, 1.0),
        ),
        LinearObjective(direction, (ObjectiveTerm("OBJECTIVE", 1.0),)),
    )
    result = sample_highs_flux_states(
        model,
        4,
        1.0,
        seed=99,
        burn_in=6,
        thinning=2,
        workers=1,
    )
    assert result.validation.valid
    assert result.validation.optimal_face_valid
    assert all(
        _values(state)["OBJECTIVE"] == pytest.approx(optimum, abs=TOLERANCE)
        for state in result.states
    )


def test_nontrivial_repository_model_samples_natively() -> None:
    before = set(sys.modules)
    model = load_ecoli_core_flux_model("biomass")
    result = sample_highs_flux_states(
        model,
        3,
        1.0,
        seed=20260904,
        burn_in=20,
        thinning=3,
        workers=1,
        max_direction_attempts=200,
    )

    assert result.sample_count == 3
    assert result.validation.valid
    reaction_order = tuple(reaction.reaction_id for reaction in model.reactions)
    assert len(reaction_order) == 95
    for state in result.states:
        assert tuple(reaction_id for reaction_id, _ in state.values) == reaction_order
        objective = _values(state)["BIOMASS_Ecoli_core_w_GAM"]
        assert objective == pytest.approx(0.8739215069684307, abs=TOLERANCE)
    assert len({tuple(state.values) for state in result.states}) > 1

    imported = set(sys.modules) - before
    assert not any(name == "cobra" or name.startswith("cobra.") for name in imported)
    assert not any(
        name == "optlang" or name.startswith("optlang.") for name in imported
    )
    assert not any(name == "mfapy" or name.startswith("mfapy.") for name in imported)


def test_native_sampling_executes_with_optional_solver_stacks_blocked() -> None:
    script = r'''
import sys
sys.modules["cobra"] = None
sys.modules["optlang"] = None
sys.modules["mfapy"] = None
import fluxemu.flux_analysis.sampling as native_sampling

assert sys.modules["cobra"] is None
assert sys.modules["optlang"] is None
assert sys.modules["mfapy"] is None

from fluxemu.model import FluxModel, FluxReaction, LinearObjective, ObjectiveTerm

model = FluxModel(
    (),
    (FluxReaction("R", (), 1.0, 1.0),),
    LinearObjective("maximise", (ObjectiveTerm("R", 1.0),)),
)
result = native_sampling.sample_highs_flux_states(
    model, 2, seed=7, burn_in=1, thinning=1, workers=1
)
assert result.validation.valid
assert len(result.states) == 2
assert sys.modules["cobra"] is None
assert sys.modules["optlang"] is None
assert sys.modules["mfapy"] is None
'''
    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
