"""Hard acceptance controls for native complete feasible-state sampling."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import importlib.util
import json
import math
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fluxemu.exceptions import AnalysisError
from fluxemu.execution import CanonicalFluxState
from fluxemu.flux_analysis import (
    FVAResult,
    compile_flux_lp,
    prepare_highs_flux_region,
    run_highs_fva_reference,
    run_highs_vffva,
    run_prepared_highs_vffva,
    sample_highs_flux_states,
    sample_prepared_flux_states,
    validate_flux_states,
)
from fluxemu.flux_analysis import highs as highs_module
from fluxemu.flux_analysis import sampling as sampling_module
from fluxemu.model import (
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    LinearObjective,
    ObjectiveTerm,
    StoichiometricTerm,
)
from fluxemu.real_model import load_ecoli_core_flux_model


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("highspy") is None,
    reason="highspy is unavailable",
)

TOLERANCE = 1e-7


def _split_path_model() -> FluxModel:
    """A bounded model with a one-dimensional optimal face."""

    metabolites = (FluxMetabolite("A", True), FluxMetabolite("B", True))
    reactions = (
        FluxReaction("SOURCE", (StoichiometricTerm("A", 1.0),), 0.0, 10.0),
        FluxReaction(
            "PATH_1",
            (StoichiometricTerm("A", -1.0), StoichiometricTerm("B", 1.0)),
            0.0,
            10.0,
        ),
        FluxReaction(
            "PATH_2",
            (StoichiometricTerm("A", -1.0), StoichiometricTerm("B", 1.0)),
            0.0,
            10.0,
        ),
        FluxReaction("BIOMASS", (StoichiometricTerm("B", -1.0),), 0.0, 10.0),
    )
    return FluxModel(
        metabolites,
        reactions,
        LinearObjective("maximise", (ObjectiveTerm("BIOMASS", 1.0),)),
    )


def _minimum_model() -> FluxModel:
    """A signed objective with a genuinely nonzero minimisation optimum."""

    return FluxModel(
        (),
        (FluxReaction("X", (), 0.0, 10.0),),
        LinearObjective("minimise", (ObjectiveTerm("X", -1.0),)),
    )


def _auxiliary_bounds_model() -> FluxModel:
    """Exercise fixed, explicitly blocked, implied-blocked, and signed fluxes."""

    metabolites = (FluxMetabolite("ISOLATED", True),)
    reactions = (
        FluxReaction("FIXED", (), 2.0, 2.0),
        FluxReaction("BLOCKED", (), 0.0, 0.0),
        FluxReaction("SIGNED", (), -4.0, 4.0),
        FluxReaction(
            "NETWORK_BLOCKED",
            (StoichiometricTerm("ISOLATED", 1.0),),
            0.0,
            10.0,
        ),
    )
    return FluxModel(
        metabolites,
        reactions,
        LinearObjective("maximise", (ObjectiveTerm("FIXED", 1.0),)),
    )


def _overflow_validation_model() -> FluxModel:
    return FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("M", 2.0),), -1.0, 1.0
            ),
            FluxReaction(
                "Y", (StoichiometricTerm("M", -2.0),), -1.0, 1.0
            ),
            FluxReaction("Q", (), 0.0, 1.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("Q", 1.0),)),
    )


def _singleton_model() -> FluxModel:
    metabolite = FluxMetabolite("A", True)
    reactions = (
        FluxReaction("IN", (StoichiometricTerm("A", 1.0),), 0.0, 10.0),
        FluxReaction("OUT", (StoichiometricTerm("A", -1.0),), 0.0, 10.0),
    )
    return FluxModel(
        (metabolite,),
        reactions,
        LinearObjective("maximise", (ObjectiveTerm("OUT", 1.0),)),
    )


def _ulp_anchor_singleton_model() -> FluxModel:
    stoichiometry = (
        (2.0, 3.0, -2.0, 3.0, 0.0, 3.0),
        (-3.0, 3.0, -1.0, 3.0, 4.0, 1.0),
    )
    objective = (0.0, 1.0, -4.0, 5.0, 4.0, -5.0)
    metabolites = (FluxMetabolite("M0", True), FluxMetabolite("M1", True))
    reactions = tuple(
        FluxReaction(
            f"R{index}",
            tuple(
                StoichiometricTerm(f"M{row}", stoichiometry[row][index])
                for row in range(2)
                if stoichiometry[row][index] != 0.0
            ),
            -1.0,
            1.0,
        )
        for index in range(6)
    )
    return FluxModel(
        metabolites,
        reactions,
        LinearObjective(
            "minimise",
            tuple(
                ObjectiveTerm(f"R{index}", coefficient)
                for index, coefficient in enumerate(objective)
            ),
        ),
    )


def _effective_zero_objective_model() -> FluxModel:
    """A nonempty declared objective whose accumulated coefficients cancel."""

    return FluxModel(
        (),
        (FluxReaction("X", (), -2.0, 2.0),),
        LinearObjective(
            "maximise",
            (ObjectiveTerm("X", 1.0), ObjectiveTerm("X", -1.0)),
        ),
    )


def _translated_zero_objective_model(offset: float) -> FluxModel:
    """The same half-unit interval translated without changing its dimension."""

    return FluxModel(
        (),
        (FluxReaction("X", (), offset, offset + 0.5),),
        LinearObjective(
            "maximise",
            (ObjectiveTerm("X", 1.0), ObjectiveTerm("X", -1.0)),
        ),
    )


def _scaled_balance_model(scale: float) -> FluxModel:
    return FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction("FIXED", (), 1.0, 1.0),
            FluxReaction(
                "X", (StoichiometricTerm("M", math.sqrt(2.0) * scale),), 0.0, 1.0
            ),
            FluxReaction(
                "Y", (StoichiometricTerm("M", -scale),), 0.0, math.sqrt(2.0)
            ),
        ),
        LinearObjective("maximise", (ObjectiveTerm("FIXED", 1.0),)),
    )


def _near_dependent_balance_model(delta: float, *, row_reduced: bool) -> FluxModel:
    metabolites = (FluxMetabolite("A", True), FluxMetabolite("B", True))
    if row_reduced:
        reaction_terms = (
            (("A", 1.0),),
            (("A", 1.0),),
            (("B", 1.0),),
            (("B", 1.0),),
        )
    else:
        reaction_terms = (
            (("A", 1.0), ("B", 1.0)),
            (("A", 1.0), ("B", 1.0)),
            (("B", delta),),
            (("B", delta),),
        )
    reactions = (FluxReaction("FIXED", (), 1.0, 1.0),) + tuple(
        FluxReaction(
            reaction_id,
            tuple(StoichiometricTerm(*term) for term in terms),
            -1.0,
            1.0,
        )
        for reaction_id, terms in zip(("X", "Y", "Z", "W"), reaction_terms)
    )
    return FluxModel(
        metabolites,
        reactions,
        LinearObjective("maximise", (ObjectiveTerm("FIXED", 1.0),)),
    )


def _mass_row_plus_constant_objective_model(direction: str) -> FluxModel:
    constant = 10.0 if direction == "maximise" else -10.0
    return FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction("FIXED", (), 1.0, 1.0),
            FluxReaction("X", (StoichiometricTerm("M", 1.0),), -1.0, 1.0),
            FluxReaction("Y", (StoichiometricTerm("M", 1.0),), -1.0, 1.0),
            FluxReaction("Z", (), 0.0, 10.0),
        ),
        LinearObjective(
            direction,
            (
                ObjectiveTerm("FIXED", constant),
                ObjectiveTerm("X", 3.0),
                ObjectiveTerm("Y", 3.0),
                ObjectiveTerm("Z", 1.0),
            ),
        ),
    )


def _dominant_mass_row_objective_model(row_coefficient: float) -> FluxModel:
    y_coefficient = math.sqrt(2.0)
    return FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction("X", (StoichiometricTerm("M", 1.0),), -1.0, 1.0),
            FluxReaction(
                "Y", (StoichiometricTerm("M", y_coefficient),), -1.0, 1.0
            ),
            FluxReaction("Z", (), -1.0, 1.0),
        ),
        LinearObjective(
            "maximise",
            (
                ObjectiveTerm("X", row_coefficient),
                ObjectiveTerm("Y", row_coefficient * y_coefficient),
                ObjectiveTerm("Z", 10.0),
            ),
        ),
    )


def _subdual_optimal_face_model() -> FluxModel:
    return FluxModel(
        (),
        (
            FluxReaction("X", (), -1.0, 1.0),
            FluxReaction("Z", (), -1.0, 1.0),
        ),
        LinearObjective(
            "maximise",
            (ObjectiveTerm("X", 1e-10), ObjectiveTerm("Z", 10.0)),
        ),
    )


def _row_space_perturbed_objective_model(
    direction: str, quotient_sign: float, row_scale: float = 1e8
) -> FluxModel:
    row = (1.0, 2.0, 3.0, 5.0)
    return FluxModel(
        (FluxMetabolite("M", True),),
        tuple(
            FluxReaction(
                reaction_id,
                (StoichiometricTerm("M", coefficient),),
                -1.0,
                1.0,
            )
            for reaction_id, coefficient in zip(("X", "Y", "Z", "W"), row)
        ),
        LinearObjective(
            direction,
            tuple(
                ObjectiveTerm(
                    reaction_id,
                    row_scale * coefficient
                    + (quotient_sign if reaction_id == "W" else 0.0),
                )
                for reaction_id, coefficient in zip(("X", "Y", "Z", "W"), row)
            ),
        ),
    )


def _near_mass_feasible_objective_state() -> CanonicalFluxState:
    return CanonicalFluxState(
        "adversarial",
        (
            ("X", -1.0),
            ("Y", -1.0),
            ("Z", -1.2500001 / 3.0),
            ("W", 0.85),
        ),
    )


def _tiny_pivot_model(epsilon: float) -> FluxModel:
    return FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "X", (StoichiometricTerm("M", epsilon),), -1.0, 1.0
            ),
            FluxReaction("Y", (StoichiometricTerm("M", 1.0),), -1.0, 1.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("X", 1.0),)),
    )


def _fixed_coordinate_scaled_balance_model(epsilon: float) -> FluxModel:
    return FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "FIX", (StoichiometricTerm("M", 1.0),), 0.0, 0.0
            ),
            FluxReaction(
                "Y", (StoichiometricTerm("M", epsilon),), -1.0, 1.0
            ),
            FluxReaction(
                "Z", (StoichiometricTerm("M", epsilon),), -1.0, 1.0
            ),
            FluxReaction("Q", (), 1.0, 1.0),
        ),
        LinearObjective("maximise", (ObjectiveTerm("Q", 1.0),)),
    )


def _values(state: CanonicalFluxState) -> dict[str, float]:
    return dict(state.values)


def _reaction_order(model: FluxModel) -> tuple[str, ...]:
    return tuple(reaction.reaction_id for reaction in model.reactions)


def _assert_valid_samples(prepared: Any, result: Any) -> None:
    assert result.sample_count == len(result.states)
    assert result.validation.valid
    independent = validate_flux_states(prepared, result.states)
    assert independent.valid
    assert independent.sample_count == len(result.states)
    expected_order = prepared.lp.reaction_ids
    assert tuple(state.sample_id for state in result.states) == tuple(
        f"sample_{index:04d}" for index in range(len(result.states))
    )
    assert all(tuple(reaction_id for reaction_id, _ in state.values) == expected_order for state in result.states)


def _record_payload(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        payload = record.to_dict()
    elif is_dataclass(record):
        payload = asdict(record)
    elif isinstance(record, dict):
        payload = record
    else:  # pragma: no cover - an explicit failure gives a useful API diagnostic
        raise AssertionError(f"record {type(record)!r} has no inspectable public payload")
    assert isinstance(payload, dict)
    return payload


def test_fraction_one_samples_distinct_complete_states_on_optimal_face() -> None:
    model = _split_path_model()
    prepared = prepare_highs_flux_region(model, 1.0)
    result = sample_prepared_flux_states(
        prepared,
        10,
        seed=17,
        burn_in=20,
        thinning=3,
        max_direction_attempts=100,
    )

    _assert_valid_samples(prepared, result)
    path_splits = []
    for state in result.states:
        values = _values(state)
        assert values["SOURCE"] == pytest.approx(10.0, abs=TOLERANCE)
        assert values["BIOMASS"] == pytest.approx(10.0, abs=TOLERANCE)
        assert values["PATH_1"] + values["PATH_2"] == pytest.approx(
            10.0, abs=TOLERANCE
        )
        path_splits.append(round(values["PATH_1"], 10))
    assert len(set(path_splits)) > 1


def test_fraction_below_one_explores_retained_region_beyond_optimal_face() -> None:
    model = _split_path_model()
    prepared = prepare_highs_flux_region(model, 0.8)
    result = sample_prepared_flux_states(
        prepared,
        12,
        seed=23,
        burn_in=20,
        thinning=3,
        max_direction_attempts=100,
    )

    _assert_valid_samples(prepared, result)
    objectives = []
    for state in result.states:
        values = _values(state)
        assert values["SOURCE"] == pytest.approx(values["BIOMASS"], abs=TOLERANCE)
        assert values["PATH_1"] + values["PATH_2"] == pytest.approx(
            values["BIOMASS"], abs=TOLERANCE
        )
        assert values["BIOMASS"] >= 8.0 - TOLERANCE
        assert values["BIOMASS"] <= 10.0 + TOLERANCE
        objectives.append(values["BIOMASS"])
    # Feasibility alone would allow an implementation stuck on the old optimum.
    assert any(value < 10.0 - 1e-5 for value in objectives)


def test_reactionwise_fva_minima_are_not_treated_as_a_correlated_flux_state() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    minima = CanonicalFluxState(
        "reactionwise-minima",
        tuple(
            (reaction_id, float(fva.ranges.loc[reaction_id, "minimum"]))
            for reaction_id in prepared.lp.reaction_ids
        ),
    )

    report = validate_flux_states(prepared, (minima,))

    assert not report.valid
    assert not report.mass_balance_valid
    assert report.max_raw_mass_balance_residual == pytest.approx(8.0)


def test_reduced_geometry_retains_the_full_mass_balance_nullspace() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    ranges = sampling_module._validate_fva_result(prepared, fva)
    geometry = sampling_module._build_reduced_geometry(prepared, ranges)

    assert geometry.basis.shape == (4, 2)
    assert geometry.equality_rank == 2
    np.testing.assert_allclose(
        sampling_module._canonical_balance_matrix(prepared.lp) @ geometry.basis,
        0.0,
        atol=1e-12,
    )


def test_fractional_minimisation_uses_nonzero_negative_objective_ceiling() -> None:
    model = _minimum_model()
    prepared = prepare_highs_flux_region(model, 0.8)
    assert prepared.fba.objective_value == pytest.approx(-10.0)
    assert prepared.retention.sense == "<="
    assert prepared.retention.bound == pytest.approx(-8.0)

    result = sample_prepared_flux_states(
        prepared,
        8,
        seed=31,
        burn_in=12,
        thinning=2,
        max_direction_attempts=100,
    )

    _assert_valid_samples(prepared, result)
    coordinates = [_values(state)["X"] for state in result.states]
    assert all(8.0 - TOLERANCE <= value <= 10.0 + TOLERANCE for value in coordinates)
    assert all(-value <= -8.0 + TOLERANCE for value in coordinates)
    assert any(value < 10.0 - 1e-5 for value in coordinates)


def test_fractional_maximisation_with_negative_coefficient_uses_floor() -> None:
    model = FluxModel(
        (),
        (FluxReaction("X", (), -10.0, 0.0),),
        LinearObjective("maximise", (ObjectiveTerm("X", -1.0),)),
    )
    prepared = prepare_highs_flux_region(model, 0.8)
    result = sample_prepared_flux_states(
        prepared, 6, seed=33, burn_in=8, thinning=2
    )
    _assert_valid_samples(prepared, result)
    coordinates = tuple(_values(state)["X"] for state in result.states)
    assert all(-10.0 - TOLERANCE <= value <= -8.0 + TOLERANCE for value in coordinates)
    assert max(coordinates) - min(coordinates) > 1e-5


@pytest.mark.parametrize(
    ("direction", "bounds"),
    [("maximise", (-10.0, -5.0)), ("minimise", (5.0, 10.0))],
)
def test_sign_incompatible_fraction_arithmetic_fails_before_fva(
    direction: str,
    bounds: tuple[float, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FluxModel(
        (),
        (FluxReaction("X", (), *bounds),),
        LinearObjective(direction, (ObjectiveTerm("X", 1.0),)),
    )

    def unexpected_fva(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("infeasible fractional arithmetic reached FVA")

    monkeypatch.setattr(sampling_module, "run_prepared_highs_vffva", unexpected_fva)
    with pytest.raises(AnalysisError, match="fraction-of-optimum arithmetic.*infeasible"):
        sample_highs_flux_states(
            model,
            2,
            seed=35,
            fraction_of_optimum=0.8,
        )


@pytest.mark.parametrize(
    ("direction", "bounds", "optimum"),
    [
        ("maximise", (-10.0, -2.0), -2.0),
        ("minimise", (2.0, 10.0), 2.0),
    ],
)
def test_wrong_sign_objective_is_a_valid_optimal_face_at_fraction_one(
    direction: str,
    bounds: tuple[float, float],
    optimum: float,
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
        5,
        seed=351,
        fraction_of_optimum=1.0,
        burn_in=8,
        thinning=2,
    )

    prepared = prepare_highs_flux_region(model, 1.0)
    _assert_valid_samples(prepared, result)
    assert all(
        _values(state)["OBJECTIVE"] == pytest.approx(optimum)
        for state in result.states
    )
    assert len({_values(state)["FREE"] for state in result.states}) > 1


@pytest.mark.parametrize(
    ("direction", "fixed_value"),
    [
        ("maximise", -5.0),
        ("maximise", -1e-12),
        ("minimise", 5.0),
        ("minimise", 1e-12),
    ],
)
def test_sign_incompatible_constant_objective_fails_before_top_level_fva(
    direction: str,
    fixed_value: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FluxModel(
        (),
        (FluxReaction("FIXED", (), fixed_value, fixed_value),),
        LinearObjective(direction, (ObjectiveTerm("FIXED", 1.0),)),
    )

    def unexpected_fva(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("infeasible constant objective reached FVA")

    monkeypatch.setattr(sampling_module, "run_prepared_highs_vffva", unexpected_fva)
    with pytest.raises(AnalysisError, match="fraction-of-optimum arithmetic.*infeasible"):
        sample_highs_flux_states(
            model,
            2,
            seed=36,
            fraction_of_optimum=0.8,
        )


@pytest.mark.parametrize(
    ("direction", "fixed_value"),
    [("maximise", 1e-12), ("minimise", -1e-12)],
)
def test_sign_compatible_small_constant_objective_keeps_full_region(
    direction: str, fixed_value: float
) -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("FIXED", (), fixed_value, fixed_value),
            FluxReaction("FREE", (), -1.0, 1.0),
        ),
        LinearObjective(direction, (ObjectiveTerm("FIXED", 1.0),)),
    )

    result = sample_highs_flux_states(
        model,
        6,
        seed=361,
        fraction_of_optimum=0.8,
        burn_in=8,
        thinning=2,
    )

    values = tuple(_values(state)["FREE"] for state in result.states)
    assert all(-1.0 - TOLERANCE <= value <= 1.0 + TOLERANCE for value in values)
    assert max(values) - min(values) > 1e-5


def test_fixed_blocked_implied_blocked_and_signed_reactions_are_preserved() -> None:
    model = _auxiliary_bounds_model()
    prepared = prepare_highs_flux_region(model, 1.0)
    result = sample_prepared_flux_states(
        prepared,
        20,
        seed=37,
        burn_in=20,
        thinning=2,
        max_direction_attempts=100,
    )

    _assert_valid_samples(prepared, result)
    signed = []
    for state in result.states:
        values = _values(state)
        assert values["FIXED"] == pytest.approx(2.0, abs=TOLERANCE)
        assert values["BLOCKED"] == pytest.approx(0.0, abs=TOLERANCE)
        assert values["NETWORK_BLOCKED"] == pytest.approx(0.0, abs=TOLERANCE)
        assert -4.0 - TOLERANCE <= values["SIGNED"] <= 4.0 + TOLERANCE
        signed.append(values["SIGNED"])
    assert min(signed) < 0.0 < max(signed)


def test_zero_dimensional_region_returns_requested_unique_ids_and_one_state() -> None:
    model = _singleton_model()
    prepared = prepare_highs_flux_region(model, 1.0)
    result = sample_prepared_flux_states(
        prepared,
        5,
        seed=41,
        burn_in=10,
        thinning=2,
        max_direction_attempts=100,
    )

    _assert_valid_samples(prepared, result)
    assert len({state.values for state in result.states}) == 1
    assert tuple(value for _, value in result.states[0].values) == pytest.approx(
        (10.0, 10.0)
    )
    provenance_text = json.dumps(_record_payload(result.provenance), sort_keys=True)
    assert '"affine_dimension": 0' in provenance_text


def test_zero_dimensional_ulp_anchor_is_repeated_without_clipping() -> None:
    model = _ulp_anchor_singleton_model()
    prepared = prepare_highs_flux_region(model, 1.0)
    cold = run_highs_fva_reference(model, 1.0)
    fast = run_prepared_highs_vffva(prepared, workers=1)
    widths = fast.ranges["maximum"] - fast.ranges["minimum"]
    result = sample_prepared_flux_states(
        prepared,
        4,
        seed=451,
        fva=fast,
        burn_in=8,
        thinning=2,
    )

    assert prepared.fba.objective_value == pytest.approx(-15.333333333333334)
    assert prepared.fba.fluxes["R0"] == -1.0000000000000002
    assert prepared.fba.fluxes["R0"] < model.reactions[0].lower_bound
    assert max(widths) <= 1e-15
    pd.testing.assert_frame_equal(cold.ranges, fast.ranges, atol=1e-15, rtol=1e-15)
    _assert_valid_samples(prepared, result)
    expected_values = tuple(
        (reaction_id, float(prepared.fba.fluxes[reaction_id]))
        for reaction_id in prepared.lp.reaction_ids
    )
    assert all(state.values == expected_values for state in result.states)
    assert all(_values(state)["R0"] < -1.0 for state in result.states)
    assert result.provenance.affine_dimension == 0
    assert 0.0 < result.validation.max_lower_bound_violation < TOLERANCE


def test_same_seed_replays_states_and_order_exactly() -> None:
    model = _split_path_model()
    kwargs = {
        "fraction_of_optimum": 0.8,
        "fva_workers": 1,
        "seed": 43,
        "burn_in": 15,
        "thinning": 2,
        "max_direction_attempts": 100,
    }
    np.random.seed(884)
    expected_global_draws = np.random.random(4)
    np.random.seed(884)
    first = sample_highs_flux_states(model, 7, **kwargs)
    observed_global_draws = np.random.random(4)
    second = sample_highs_flux_states(model, 7, **kwargs)

    np.testing.assert_array_equal(observed_global_draws, expected_global_draws)
    assert first.states == second.states
    assert first.provenance == second.provenance
    pd.testing.assert_frame_equal(first.to_frame(), second.to_frame(), check_exact=True)


@pytest.mark.parametrize("fraction", [1.0, 0.8])
def test_effective_zero_objective_keeps_the_bounded_feasible_region(
    fraction: float,
) -> None:
    model = _effective_zero_objective_model()
    prepared = prepare_highs_flux_region(model, fraction)
    assert prepared.fba.objective_value == pytest.approx(0.0)
    assert prepared.retention.bound == pytest.approx(0.0)

    result = sample_prepared_flux_states(
        prepared,
        8,
        seed=45,
        burn_in=10,
        thinning=2,
        max_direction_attempts=100,
    )

    _assert_valid_samples(prepared, result)
    coordinates = [_values(state)["X"] for state in result.states]
    assert all(-2.0 - TOLERANCE <= value <= 2.0 + TOLERANCE for value in coordinates)
    assert max(coordinates) - min(coordinates) > 1e-5


def test_zero_active_objective_scale_ignores_only_roundoff_sign_noise() -> None:
    model = FluxModel(
        (FluxMetabolite("M", True),),
        (
            FluxReaction(
                "R0", (StoichiometricTerm("M", 4.0),), 0.0, 0.0
            ),
            FluxReaction(
                "R1", (StoichiometricTerm("M", 4.0),), -1.0, 1.0
            ),
            FluxReaction(
                "R2", (StoichiometricTerm("M", -3.0),), -1.0, 1.0
            ),
        ),
        LinearObjective(
            "maximise",
            (
                ObjectiveTerm("R0", 4.0),
                ObjectiveTerm("R1", -4.0),
                ObjectiveTerm("R2", 3.0),
            ),
        ),
    )
    prepared = prepare_highs_flux_region(model, 0.3)
    cold = run_highs_fva_reference(model, 0.3)
    fast = run_prepared_highs_vffva(prepared, workers=1)
    result = sample_prepared_flux_states(
        prepared,
        6,
        seed=452,
        fva=fast,
        burn_in=8,
        thinning=2,
    )

    assert prepared.fba.objective_value == pytest.approx(0.0)
    assert prepared.lp.effective_objective_coefficients == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    assert prepared.retention.bound == pytest.approx(0.0)
    pd.testing.assert_frame_equal(cold.ranges, fast.ranges, atol=1e-10, rtol=1e-10)
    assert tuple(fast.ranges.loc["R0"]) == pytest.approx((0.0, 0.0))
    assert tuple(fast.ranges.loc["R1"]) == pytest.approx((-0.75, 0.75))
    assert tuple(fast.ranges.loc["R2"]) == pytest.approx((-1.0, 1.0))
    _assert_valid_samples(prepared, result)
    values = tuple(_values(state) for state in result.states)
    assert all(
        4.0 * item["R1"] - 3.0 * item["R2"] == pytest.approx(
            0.0,
            abs=TOLERANCE,
        )
        for item in values
    )
    assert len({round(item["R2"], 10) for item in values}) > 1
    assert result.validation.max_stable_retained_objective_violation == pytest.approx(
        0.0
    )
    assert result.validation.max_objective_conditioning_discrepancy < 1e-12


def test_duplicate_objective_and_stoichiometric_terms_use_fsum_in_any_order() -> None:
    def cancellation_model(reverse: bool) -> FluxModel:
        cancelling = (
            (-1e16, 1.0, 1e16) if reverse else (1e16, 1.0, -1e16)
        )
        return FluxModel(
            (FluxMetabolite("M", True),),
            (
                FluxReaction(
                    "X",
                    tuple(
                        StoichiometricTerm("M", coefficient)
                        for coefficient in cancelling
                    ),
                    0.0,
                    2.0,
                ),
                FluxReaction(
                    "Y", (StoichiometricTerm("M", -1.0),), 0.0, 2.0
                ),
                FluxReaction("Z", (), 0.0, 2.0),
            ),
            LinearObjective(
                "maximise",
                tuple(
                    ObjectiveTerm("Z", coefficient) for coefficient in cancelling
                ),
            ),
        )

    forward = compile_flux_lp(cancellation_model(False))
    reverse = compile_flux_lp(cancellation_model(True))

    assert forward.coefficients == reverse.coefficients == (1.0, -1.0)
    assert forward.objective_coefficients == reverse.objective_coefficients == (
        0.0,
        0.0,
        1.0,
    )
    assert forward.fingerprint == reverse.fingerprint
    assert prepare_highs_flux_region(
        cancellation_model(False), 1.0
    ).fba.objective_value == pytest.approx(2.0)
    assert prepare_highs_flux_region(
        cancellation_model(True), 1.0
    ).fba.objective_value == pytest.approx(2.0)


def test_fva_face_detection_is_invariant_to_coordinate_translation() -> None:
    results = []
    for offset in (0.0, 1e10):
        result = sample_highs_flux_states(
            _translated_zero_objective_model(offset),
            6,
            seed=46,
            burn_in=8,
            thinning=2,
        )
        assert result.provenance.affine_dimension == 1
        assert result.provenance.center_radius == pytest.approx(0.25, abs=1e-7)
        shifted = tuple(_values(state)["X"] - offset for state in result.states)
        assert max(shifted) - min(shifted) > 1e-3
        results.append(shifted)
    assert results[1] == pytest.approx(results[0], abs=3e-6)


def test_sampling_and_validation_are_invariant_to_balance_row_scaling() -> None:
    sampled_x = []
    for scale in (1.0, 1e-12, 1e-200):
        model = _scaled_balance_model(scale)
        prepared = prepare_highs_flux_region(model, 1.0)
        result = sample_prepared_flux_states(
            prepared,
            5,
            seed=48,
            burn_in=8,
            thinning=2,
        )
        _assert_valid_samples(prepared, result)
        assert result.validation.max_mass_balance_residual <= TOLERANCE
        sampled_x.append(tuple(_values(state)["X"] for state in result.states))
    assert all(
        values == pytest.approx(sampled_x[0], abs=1e-7)
        for values in sampled_x[1:]
    )


@pytest.mark.parametrize("epsilon", [1e-8, 1e-200])
def test_fixed_coordinate_is_eliminated_before_scaled_hull_conditioning(
    epsilon: float,
) -> None:
    model = _fixed_coordinate_scaled_balance_model(epsilon)
    prepared = prepare_highs_flux_region(model, 1.0)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    result = sample_prepared_flux_states(
        prepared,
        6,
        seed=480,
        fva=fva,
        burn_in=8,
        thinning=2,
    )

    assert tuple(fva.ranges.loc["FIX"]) == pytest.approx((0.0, 0.0))
    assert tuple(fva.ranges.loc["Y"]) == pytest.approx((-1.0, 1.0))
    assert tuple(fva.ranges.loc["Z"]) == pytest.approx((-1.0, 1.0))
    _assert_valid_samples(prepared, result)
    states = tuple(_values(state) for state in result.states)
    assert all(values["FIX"] == pytest.approx(0.0) for values in states)
    assert all(values["Y"] + values["Z"] == pytest.approx(0.0) for values in states)
    assert len({round(values["Y"], 10) for values in states}) > 1


@pytest.mark.parametrize("delta", [1e-10, 1e-15])
def test_near_dependent_mass_balance_rows_fail_closed(delta: float) -> None:
    with pytest.raises(
        AnalysisError,
        match=(
            "balanced-metabolite.*ill-conditioned|ambiguous rank|"
            "exact independent direction below numerical resolution"
        ),
    ):
        prepare_highs_flux_region(
            _near_dependent_balance_model(delta, row_reduced=False), 1.0
        )


def test_explicit_row_reduction_of_near_dependent_model_samples_correctly() -> None:
    model = _near_dependent_balance_model(1e-10, row_reduced=True)
    prepared = prepare_highs_flux_region(model, 1.0)
    result = sample_prepared_flux_states(
        prepared,
        6,
        seed=481,
        burn_in=8,
        thinning=2,
    )

    _assert_valid_samples(prepared, result)
    observed = []
    for state in result.states:
        values = _values(state)
        assert values["X"] + values["Y"] == pytest.approx(0.0, abs=TOLERANCE)
        assert values["Z"] + values["W"] == pytest.approx(0.0, abs=TOLERANCE)
        observed.append((round(values["X"], 10), round(values["Z"], 10)))
    assert len(set(observed)) > 1


@pytest.mark.parametrize(
    (
        "direction",
        "fraction",
        "expected_optimum",
        "expected_constant",
        "expected_effective_bound",
        "expected_z_range",
    ),
    [
        ("maximise", 1.0, 20.0, 10.0, 10.0, (10.0, 10.0)),
        ("maximise", 0.8, 20.0, 10.0, 6.0, (6.0, 10.0)),
        ("minimise", 1.0, -10.0, -10.0, 0.0, (0.0, 0.0)),
        ("minimise", 0.8, -10.0, -10.0, 2.0, (0.0, 2.0)),
    ],
)
def test_mass_row_objective_and_fixed_constant_preserve_expected_z_region(
    direction: str,
    fraction: float,
    expected_optimum: float,
    expected_constant: float,
    expected_effective_bound: float,
    expected_z_range: tuple[float, float],
) -> None:
    model = _mass_row_plus_constant_objective_model(direction)
    prepared = prepare_highs_flux_region(model, fraction)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    result = sample_prepared_flux_states(
        prepared,
        5,
        seed=482,
        fva=fva,
        burn_in=8,
        thinning=2,
    )

    assert prepared.fba.objective_value == pytest.approx(expected_optimum)
    assert prepared.retention.bound == pytest.approx(fraction * expected_optimum)
    assert prepared.lp.objective_constant == pytest.approx(expected_constant)
    assert prepared.retention.effective_bound == pytest.approx(
        expected_effective_bound
    )
    assert tuple(fva.ranges.loc["Z"]) == pytest.approx(expected_z_range)
    assert result.provenance.declared_objective_scale == pytest.approx(10.0)
    assert result.provenance.effective_objective_scale == pytest.approx(
        max(abs(value) for value in prepared.lp.effective_objective_coefficients)
    )
    assert result.provenance.effective_objective_scale > 0.0
    assert result.provenance.objective_constant == pytest.approx(expected_constant)
    assert result.provenance.effective_objective_bound == pytest.approx(
        expected_effective_bound
    )
    _assert_valid_samples(prepared, result)
    for state in result.states:
        values = _values(state)
        assert values["X"] + values["Y"] == pytest.approx(0.0, abs=TOLERANCE)
        assert expected_z_range[0] - TOLERANCE <= values["Z"]
        assert values["Z"] <= expected_z_range[1] + TOLERANCE


def test_subdual_objective_term_fails_closed_for_exact_optimal_face_path() -> None:
    model = _subdual_optimal_face_model()
    operations = (
        lambda: prepare_highs_flux_region(model, 1.0),
        lambda: run_highs_fva_reference(model, 1.0),
        lambda: run_highs_vffva(model, 1.0, workers=1),
        lambda: sample_highs_flux_states(
            model,
            2,
            seed=484,
            fraction_of_optimum=1.0,
        ),
    )

    for operation in operations:
        with pytest.raises(AnalysisError) as caught:
            operation()
        message = str(caught.value).lower()
        assert all(token in message for token in ("objective", "optimal", "face"))
        assert "resolution" in message or "ambiguous" in message


def test_dominant_mass_row_component_fails_for_unresolved_optimal_face() -> None:
    with pytest.raises(AnalysisError) as caught:
        prepare_highs_flux_region(_dominant_mass_row_objective_model(1e8), 1.0)

    message = str(caught.value).lower()
    assert all(token in message for token in ("objective", "optimal", "face"))
    assert "resolution" in message or "ambiguous" in message


def test_dominant_mass_row_component_preserves_fractional_objective_region() -> None:
    fraction = 0.8
    model = _dominant_mass_row_objective_model(1e8)
    prepared = prepare_highs_flux_region(model, fraction)
    cold = run_highs_fva_reference(model, fraction)
    fast = run_prepared_highs_vffva(prepared, workers=1)
    result = sample_prepared_flux_states(
        prepared,
        5,
        seed=483,
        fva=fast,
        burn_in=8,
        thinning=2,
    )

    assert prepared.fba.objective_value == pytest.approx(10.0)
    assert prepared.fba.fluxes["Z"] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(cold.ranges, fast.ranges, atol=1e-8, rtol=1e-8)
    assert tuple(fast.ranges.loc["X"]) == pytest.approx((-1.0, 1.0))
    assert tuple(fast.ranges.loc["Y"]) == pytest.approx(
        (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    )
    assert tuple(fast.ranges.loc["Z"]) == pytest.approx((fraction, 1.0))
    assert result.provenance.effective_objective_scale == pytest.approx(
        max(abs(value) for value in prepared.lp.effective_objective_coefficients)
    )
    _assert_valid_samples(prepared, result)
    states = tuple(_values(state) for state in result.states)
    assert all(
        values["X"] + math.sqrt(2.0) * values["Y"] == pytest.approx(0.0)
        for values in states
    )
    assert all(fraction - TOLERANCE <= values["Z"] <= 1.0 + TOLERANCE for values in states)
    assert len({round(values["X"], 10) for values in states}) > 1
    diagnostics = result.validation.diagnostics
    assert any(item.objective_conditioning_discrepancy > 0.0 for item in diagnostics)
    for item in diagnostics:
        assert item.objective_value is not None
        assert item.stable_objective_value is not None
        assert item.objective_conditioning_discrepancy == pytest.approx(
            abs(item.objective_value - item.stable_objective_value)
        )
        assert item.normalized_objective_conditioning_discrepancy <= TOLERANCE
    assert result.validation.max_objective_conditioning_discrepancy == max(
        item.objective_conditioning_discrepancy for item in diagnostics
    )


@pytest.mark.parametrize("row_coefficient", [1e11, 1e16])
def test_oversized_mass_row_component_fails_on_direct_objective_discrepancy(
    row_coefficient: float,
) -> None:
    with pytest.raises(AnalysisError) as caught:
        prepare_highs_flux_region(
            _dominant_mass_row_objective_model(row_coefficient), 1.0
        )

    message = str(caught.value).lower()
    assert all(token in message for token in ("declared", "quotient", "discrepancy"))


def test_masked_random_objective_fails_on_direct_objective_discrepancy() -> None:
    row_vectors = (
        (
            -0.16677573400198623,
            0.7497655536581233,
            0.3914960373481072,
            -0.3936374712970443,
        ),
        (
            -1.6190718201181646,
            0.11387519414201226,
            1.0790096392361133,
            -1.1975882890341571,
        ),
        (
            0.638759381675977,
            0.33154169513369264,
            0.7501482632323336,
            0.8719754997190072,
        ),
    )
    objective = (
        8.352834040402516e18,
        -4.802874765926683e18,
        -2.659533714196261e18,
        1.006774432656613e19,
    )
    metabolites = tuple(
        FluxMetabolite(f"M{index}", True) for index in range(len(row_vectors))
    )
    reactions = tuple(
        FluxReaction(
            f"V{column}",
            tuple(
                StoichiometricTerm(f"M{row}", row_vectors[row][column])
                for row in range(len(row_vectors))
            ),
            -1.0,
            1.0,
        )
        for column in range(4)
    )
    model = FluxModel(
        metabolites,
        reactions,
        LinearObjective(
            "maximise",
            tuple(
                ObjectiveTerm(f"V{index}", coefficient)
                for index, coefficient in enumerate(objective)
            ),
        ),
    )

    with pytest.raises(AnalysisError) as caught:
        prepare_highs_flux_region(model, 1.0)

    message = str(caught.value).lower()
    assert all(token in message for token in ("declared", "quotient", "discrepancy"))


@pytest.mark.parametrize("direction", ["maximise", "minimise"])
@pytest.mark.parametrize("quotient_sign", [1.0, -1.0])
@pytest.mark.parametrize("fraction", [1.0, 0.8])
def test_exact_row_space_quotient_preserves_signed_w_objective_region(
    direction: str, quotient_sign: float, fraction: float
) -> None:
    model = _row_space_perturbed_objective_model(direction, quotient_sign)
    prepared = prepare_highs_flux_region(model, fraction)
    cold = run_highs_fva_reference(model, fraction)
    fast = run_prepared_highs_vffva(prepared, workers=1)
    result = sample_prepared_flux_states(
        prepared,
        5,
        seed=485,
        fva=fast,
        burn_in=8,
        thinning=2,
    )

    maximize = direction == "maximise"
    optimal_w = quotient_sign if maximize else -quotient_sign
    expected_objective = 1.0 if maximize else -1.0
    expected_w_range = (
        (fraction, 1.0) if optimal_w > 0.0 else (-1.0, -fraction)
    )
    assert prepared.fba.objective_value == pytest.approx(expected_objective)
    assert prepared.fba.fluxes["W"] == pytest.approx(optimal_w)
    pd.testing.assert_frame_equal(cold.ranges, fast.ranges, atol=1e-8, rtol=1e-8)
    assert tuple(fast.ranges.loc["W"]) == pytest.approx(expected_w_range)
    if fraction == 1.0:
        expected_face = (
            {"X": (-1.0, 0.0), "Y": (-1.0, -0.5), "Z": (-1.0, -2.0 / 3.0)}
            if optimal_w > 0.0
            else {"X": (0.0, 1.0), "Y": (0.5, 1.0), "Z": (2.0 / 3.0, 1.0)}
        )
        for reaction_id, expected_range in expected_face.items():
            assert tuple(fast.ranges.loc[reaction_id]) == pytest.approx(
                expected_range, abs=1e-8
            )

    _assert_valid_samples(prepared, result)
    states = tuple(_values(state) for state in result.states)
    for values in states:
        assert (
            values["X"] + 2.0 * values["Y"] + 3.0 * values["Z"] + 5.0 * values["W"]
        ) == pytest.approx(0.0, abs=TOLERANCE)
        assert expected_w_range[0] - TOLERANCE <= values["W"]
        assert values["W"] <= expected_w_range[1] + TOLERANCE
    assert len({state.values for state in result.states}) > 1


@pytest.mark.parametrize("epsilon", [1e-10, 1e-12])
def test_tiny_pivot_optimal_face_does_not_erase_x_objective(epsilon: float) -> None:
    model = _tiny_pivot_model(epsilon)
    prepared = prepare_highs_flux_region(model, 1.0)
    cold = run_highs_fva_reference(model, 1.0)
    fast = run_prepared_highs_vffva(prepared, workers=1)
    result = sample_prepared_flux_states(prepared, 2, seed=486, fva=fast)
    assert prepared.fba.objective_value == pytest.approx(1.0)
    assert prepared.fba.fluxes["X"] == pytest.approx(1.0)
    assert abs(epsilon * prepared.fba.fluxes["X"] + prepared.fba.fluxes["Y"]) <= TOLERANCE
    pd.testing.assert_frame_equal(cold.ranges, fast.ranges, atol=1e-9, rtol=1e-8)
    assert tuple(fast.ranges.loc["X"]) == pytest.approx((1.0, 1.0))
    _assert_valid_samples(prepared, result)
    for state in result.states:
        values = _values(state)
        assert values["X"] == pytest.approx(1.0)
        assert abs(epsilon * values["X"] + values["Y"]) <= TOLERANCE


@pytest.mark.parametrize("epsilon", [1e-10, 1e-12])
def test_tiny_pivot_fractional_fva_retains_x_objective_region(
    epsilon: float,
) -> None:
    model = _tiny_pivot_model(epsilon)
    cold = run_highs_fva_reference(model, 0.8)
    fast = run_highs_vffva(model, 0.8, workers=1)
    pd.testing.assert_frame_equal(cold.ranges, fast.ranges, atol=1e-9, rtol=1e-8)
    assert tuple(fast.ranges.loc["X"]) == pytest.approx((0.8, 1.0))
    for endpoint in ("minimum", "maximum"):
        assert abs(
            epsilon * fast.ranges.loc["X", endpoint]
            + fast.ranges.loc["Y", endpoint]
        ) <= TOLERANCE


@pytest.mark.parametrize("coefficient", [1.0, 1e-12, 1e8])
def test_objective_scaling_preserves_fba_fva_and_optimal_face(
    coefficient: float,
) -> None:
    model = FluxModel(
        (),
        (FluxReaction("X", (), 0.0, 10.0),),
        LinearObjective("maximise", (ObjectiveTerm("X", coefficient),)),
    )
    prepared = prepare_highs_flux_region(model, 1.0)
    fva = run_prepared_highs_vffva(prepared, workers=1)
    result = sample_prepared_flux_states(prepared, 3, seed=49, fva=fva)

    assert prepared.fba.fluxes["X"] == pytest.approx(10.0)
    assert prepared.fba.objective_value == pytest.approx(10.0 * coefficient)
    assert tuple(fva.ranges.loc["X"]) == pytest.approx((10.0, 10.0))
    assert all(_values(state)["X"] == pytest.approx(10.0) for state in result.states)
    assert result.provenance.declared_objective_scale == pytest.approx(
        abs(coefficient)
    )
    assert result.provenance.effective_objective_scale == pytest.approx(
        max(abs(value) for value in prepared.lp.effective_objective_coefficients)
    )
    assert result.provenance.effective_objective_scale > 0.0


def test_small_scaled_objective_fractional_region_is_not_dropped() -> None:
    model = FluxModel(
        (),
        (FluxReaction("X", (), 0.0, 10.0),),
        LinearObjective("maximise", (ObjectiveTerm("X", 1e-12),)),
    )
    prepared = prepare_highs_flux_region(model, 0.8)
    result = sample_prepared_flux_states(
        prepared, 6, seed=50, burn_in=8, thinning=2
    )
    _assert_valid_samples(prepared, result)
    values = tuple(_values(state)["X"] for state in result.states)
    assert all(8.0 - TOLERANCE <= value <= 10.0 + TOLERANCE for value in values)
    assert max(values) - min(values) > 1e-5


def test_ordered_frame_and_sampling_provenance_are_complete() -> None:
    model = _split_path_model()
    prepared = prepare_highs_flux_region(model, 0.8)
    result = sample_prepared_flux_states(
        prepared,
        4,
        seed=47,
        burn_in=11,
        thinning=3,
        max_direction_attempts=29,
    )
    _assert_valid_samples(prepared, result)

    frame = result.to_frame()
    assert tuple(frame.columns) == _reaction_order(model)
    assert tuple(frame.index) == tuple(state.sample_id for state in result.states)
    assert frame.index.name == "sample_id"
    for state in result.states:
        assert tuple(frame.loc[state.sample_id]) == pytest.approx(
            tuple(value for _, value in state.values)
        )

    payload = _record_payload(result.provenance)
    text = json.dumps(payload, sort_keys=True)
    lowered = text.lower()
    assert "hit-and-run" in lowered
    assert "pcg64" in lowered
    assert prepared.lp.fingerprint in text
    assert _reaction_order(model) == tuple(payload["reaction_order"])
    assert payload["seed"] == 47
    assert payload["fraction_of_optimum"] == pytest.approx(0.8)
    assert payload["biological_optimum"] == pytest.approx(10.0)
    assert payload["retained_objective_bound"] == pytest.approx(8.0)
    assert payload["objective_direction"] == "max"
    assert payload["declared_objective_scale"] == pytest.approx(1.0)
    assert payload["effective_objective_scale"] == pytest.approx(
        max(abs(value) for value in prepared.lp.effective_objective_coefficients)
    )
    assert payload["effective_objective_scale"] > 0.0
    assert payload["objective_constant"] == pytest.approx(0.0)
    assert payload["effective_objective_bound"] == pytest.approx(8.0)
    assert len(payload["fva_ranges_sha256"]) == 64
    assert payload["retained_objective_sense"] == ">="
    assert payload["burn_in"] == 11
    assert payload["thinning"] == 3
    assert payload["max_direction_attempts"] == 29
    assert payload["equality_rank"] > 0
    assert payload["affine_dimension"] > 0
    assert payload["center_method"] == "chebyshev-center"
    assert payload["center_radius"] > 0.0
    assert 0 <= payload["self_loop_steps"] <= payload["accepted_chain_steps"]
    assert payload["bound_tolerance"] == pytest.approx(TOLERANCE)
    assert payload["mass_balance_tolerance"] == pytest.approx(TOLERANCE)
    assert payload["retained_objective_tolerance"] == pytest.approx(TOLERANCE)
    assert payload["reduced_direction_tolerance"] == pytest.approx(
        sampling_module.REDUCED_DIRECTION_TOLERANCE
    )
    assert payload["center_radius_tolerance"] == pytest.approx(
        sampling_module.CENTER_RADIUS_TOLERANCE
    )
    assert payload["numpy_version"]
    assert payload["highs_version"]


def test_validator_rejects_declared_objective_disagreement_on_near_mass_state() -> None:
    prepared = prepare_highs_flux_region(
        _row_space_perturbed_objective_model(
            "maximise",
            1.0,
            row_scale=1e6,
        ),
        0.8,
    )
    report = validate_flux_states(
        prepared,
        (_near_mass_feasible_objective_state(),),
    )

    assert not report.valid
    assert report.bounds_valid
    assert report.mass_balance_valid
    assert not report.retained_objective_valid
    assert report.max_mass_balance_residual < TOLERANCE
    diagnostic = report.diagnostics[0]
    assert diagnostic.objective_value == pytest.approx(0.75, abs=1e-8)
    assert diagnostic.stable_objective_value == pytest.approx(0.85, abs=1e-7)
    assert diagnostic.objective_conditioning_discrepancy == pytest.approx(
        0.1,
        abs=1e-7,
    )
    assert diagnostic.raw_retained_objective_violation == pytest.approx(
        0.05,
        abs=1e-8,
    )
    assert diagnostic.retained_objective_violation > TOLERANCE
    assert diagnostic.stable_retained_objective_violation == pytest.approx(0.0)
    lowered = " ".join(report.errors).lower()
    assert all(
        token in lowered
        for token in (
            "declared",
            "quotient",
            "discrepancy",
            "retained objective",
            "raw_violation",
        )
    )


def test_validator_detects_retention_hidden_by_large_fixed_objective_constant() -> None:
    fixed_constant = float(2**50)
    model = FluxModel(
        (),
        (
            FluxReaction("Q", (), 1.0, 1.0),
            FluxReaction("Z", (), 0.0, 1.0),
        ),
        LinearObjective(
            "maximise",
            (
                ObjectiveTerm("Q", fixed_constant),
                ObjectiveTerm("Z", 1.0),
            ),
        ),
    )
    prepared = prepare_highs_flux_region(model, 1.0)
    cold = run_highs_fva_reference(model, 1.0)
    fast = run_prepared_highs_vffva(prepared, workers=1)
    states = (
        CanonicalFluxState("near", (("Q", 1.0), ("Z", 0.925))),
        CanonicalFluxState("zero", (("Q", 1.0), ("Z", 0.0))),
    )
    report = validate_flux_states(prepared, states)

    assert prepared.lp.objective_constant == fixed_constant
    assert prepared.retention.effective_bound == pytest.approx(1.0)
    pd.testing.assert_frame_equal(cold.ranges, fast.ranges, atol=1e-12, rtol=1e-12)
    assert tuple(fast.ranges.loc["Z"]) == pytest.approx((1.0, 1.0))
    assert not report.valid
    assert not report.retained_objective_valid
    near, zero = report.diagnostics
    assert near.objective_value == near.stable_objective_value
    assert near.effective_objective_value == pytest.approx(0.925)
    assert near.raw_retained_objective_violation == pytest.approx(0.0)
    assert near.stable_retained_objective_violation == pytest.approx(0.075)
    assert zero.effective_objective_value == pytest.approx(0.0)
    assert zero.retained_objective_violation == pytest.approx(1.0)
    assert zero.stable_retained_objective_violation == pytest.approx(1.0)
    assert report.max_stable_retained_objective_violation == pytest.approx(1.0)
    assert all(
        "conditioned retained objective" in " ".join(item.errors).lower()
        for item in report.diagnostics
    )


@pytest.mark.parametrize(
    ("case", "pairs", "expected_tokens"),
    [
        (
            "reordered",
            (("BIOMASS", 9.0), ("PATH_2", 5.0), ("PATH_1", 4.0), ("SOURCE", 9.0)),
            ("order",),
        ),
        (
            "duplicate",
            (("SOURCE", 9.0), ("PATH_1", 4.0), ("PATH_1", 5.0), ("BIOMASS", 9.0)),
            ("duplicate", "path_1"),
        ),
        (
            "missing",
            (("SOURCE", 9.0), ("PATH_1", 4.0), ("BIOMASS", 9.0)),
            ("missing", "path_2"),
        ),
        (
            "unexpected",
            (("SOURCE", 9.0), ("PATH_1", 4.0), ("UNKNOWN", 5.0), ("BIOMASS", 9.0)),
            ("unexpected", "unknown"),
        ),
        (
            "nonfinite",
            (("SOURCE", 9.0), ("PATH_1", math.nan), ("PATH_2", 5.0), ("BIOMASS", 9.0)),
            ("finite", "path_1"),
        ),
        (
            "boolean",
            (("SOURCE", 9.0), ("PATH_1", True), ("PATH_2", 5.0), ("BIOMASS", 9.0)),
            ("numeric", "path_1"),
        ),
        (
            "nonnumeric",
            (("SOURCE", 9.0), ("PATH_1", "four"), ("PATH_2", 5.0), ("BIOMASS", 9.0)),
            ("numeric", "path_1"),
        ),
        (
            "lower",
            (("SOURCE", 9.0), ("PATH_1", -1.0), ("PATH_2", 10.0), ("BIOMASS", 9.0)),
            ("lower", "path_1"),
        ),
        (
            "upper",
            (("SOURCE", 11.0), ("PATH_1", 5.0), ("PATH_2", 6.0), ("BIOMASS", 11.0)),
            ("upper", "source"),
        ),
        (
            "mass_balance",
            (("SOURCE", 9.0), ("PATH_1", 4.25), ("PATH_2", 5.0), ("BIOMASS", 9.0)),
            ("balance", "metabolite", "a"),
        ),
        (
            "objective",
            (("SOURCE", 7.5), ("PATH_1", 3.0), ("PATH_2", 4.5), ("BIOMASS", 7.5)),
            ("objective",),
        ),
    ],
)
def test_independent_validator_localizes_corrupt_complete_states(
    case: str,
    pairs: tuple[tuple[str, float], ...],
    expected_tokens: tuple[str, ...],
) -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    state = CanonicalFluxState(f"bad-{case}", pairs)

    report = validate_flux_states(prepared, (state,))

    assert not report.valid
    text = json.dumps(report.to_dict(), sort_keys=True, default=str).lower()
    assert all(token in text for token in expected_tokens)


def test_independent_validator_rejects_duplicate_sample_ids() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    values = (
        ("SOURCE", 9.0),
        ("PATH_1", 4.0),
        ("PATH_2", 5.0),
        ("BIOMASS", 9.0),
    )
    states = (
        CanonicalFluxState("duplicate", values),
        CanonicalFluxState("duplicate", values),
    )

    report = validate_flux_states(prepared, states)

    assert not report.valid
    text = json.dumps(report.to_dict(), sort_keys=True, default=str).lower()
    assert "duplicate" in text and "sample" in text


def test_validator_reports_huge_integer_flux_without_raising() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    state = CanonicalFluxState(
        "huge-integer",
        (
            ("SOURCE", 10**10000),
            ("PATH_1", 4.0),
            ("PATH_2", 5.0),
            ("BIOMASS", 9.0),
        ),
    )

    report = validate_flux_states(prepared, (state,))

    assert not report.valid
    assert not report.finite_values_valid
    assert report.sample_count == 1
    assert report.diagnostics[0].sample_id == "huge-integer"
    assert not report.diagnostics[0].finite_values_valid
    assert any(
        "non-finite" in error and "SOURCE" in error for error in report.errors
    )


def test_validator_returns_infinite_overflow_diagnostics_without_raising() -> None:
    prepared = prepare_highs_flux_region(_overflow_validation_model(), 1.0)
    state = CanonicalFluxState(
        "overflow",
        (("X", 1e308), ("Y", 1e308), ("Q", 1.0)),
    )

    report = validate_flux_states(prepared, (state,))

    assert not report.valid
    assert report.finite_values_valid
    assert not report.bounds_valid
    assert not report.mass_balance_valid
    assert report.retained_objective_valid
    assert report.max_upper_bound_violation == pytest.approx(1e308)
    assert math.isinf(report.max_raw_mass_balance_residual)
    assert math.isinf(report.max_mass_balance_residual)
    assert math.isinf(report.max_conditioned_row_space_residual)
    diagnostic = report.diagnostics[0]
    assert diagnostic.finite_values_valid
    assert diagnostic.upper_bound_reaction_id == "X"
    assert math.isinf(diagnostic.max_raw_mass_balance_residual)
    assert math.isinf(diagnostic.max_mass_balance_residual)
    assert math.isinf(diagnostic.conditioned_row_space_residual)
    assert any("residual=inf" in error for error in diagnostic.errors)


def test_validator_rejects_finite_state_when_objective_arithmetic_overflows() -> None:
    model = FluxModel(
        (),
        (FluxReaction("X", (), -1.0, 1.0),),
        LinearObjective("maximise", (ObjectiveTerm("X", 2.0),)),
    )
    prepared = prepare_highs_flux_region(model, 1.0)
    state = CanonicalFluxState("objective-overflow", (("X", 1e308),))

    report = validate_flux_states(prepared, (state,))

    assert not report.valid
    assert report.finite_values_valid
    assert not report.bounds_valid
    assert not report.retained_objective_valid
    assert math.isinf(report.diagnostics[0].objective_value)
    assert math.isinf(report.max_retained_objective_violation)


def test_validator_fails_closed_on_nonfinite_raw_mass_arithmetic() -> None:
    model = FluxModel(
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
    prepared = prepare_highs_flux_region(model, 1.0)
    state = CanonicalFluxState(
        "mass-overflow",
        (("X", 2.0), ("Y", 2.0), ("Q", 1.0)),
    )

    report = validate_flux_states(prepared, (state,))

    assert not report.valid
    assert report.bounds_valid
    assert not report.mass_balance_valid
    assert math.isinf(report.max_raw_mass_balance_residual)
    assert math.isinf(report.max_mass_balance_residual)
    assert any("non-finite" in error.lower() for error in report.errors)


def test_supplied_fva_is_used_without_running_fva_again(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    supplied = run_prepared_highs_vffva(prepared, workers=1)
    expected_ranges = supplied.ranges.copy(deep=True)

    def unexpected_preparation(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("prepared sampling repeated model preparation")

    monkeypatch.setattr(
        sampling_module, "prepare_highs_flux_region", unexpected_preparation
    )
    monkeypatch.setattr(
        sampling_module, "run_prepared_highs_vffva", unexpected_preparation
    )
    result = sample_prepared_flux_states(
        prepared,
        3,
        seed=59,
        fva=supplied,
        burn_in=3,
        thinning=1,
        max_direction_attempts=20,
    )

    _assert_valid_samples(prepared, result)
    pd.testing.assert_frame_equal(supplied.ranges, expected_ranges, check_exact=True)


def test_supplied_fva_with_mismatched_retained_region_is_rejected() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    different_region = prepare_highs_flux_region(_split_path_model(), 1.0)
    mismatched = run_prepared_highs_vffva(different_region, workers=1)

    with pytest.raises(AnalysisError, match="FVA.*metadata.*prepared region"):
        sample_prepared_flux_states(prepared, 2, seed=61, fva=mismatched)


def test_supplied_fva_is_bound_to_the_compiled_model_fingerprint() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    supplied = run_prepared_highs_vffva(prepared, workers=1)
    forged = replace(supplied, model_fingerprint="0" * 64)

    with pytest.raises(AnalysisError, match="FVA model fingerprint.*prepared"):
        sample_prepared_flux_states(prepared, 2, seed=63, fva=forged)


def test_tiny_nonzero_fva_width_is_named_as_numerically_degenerate() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    supplied = run_prepared_highs_vffva(prepared, workers=1)
    narrowed = supplied.ranges.copy()
    fba_value = float(prepared.fba.fluxes["PATH_1"])
    if fba_value <= 5.0:
        narrowed.loc["PATH_1", "minimum"] = fba_value
        narrowed.loc["PATH_1", "maximum"] = fba_value + 5e-9
    else:
        narrowed.loc["PATH_1", "minimum"] = fba_value - 5e-9
        narrowed.loc["PATH_1", "maximum"] = fba_value
    narrowed_result = replace(
        supplied,
        ranges=narrowed,
        ranges_sha256=sampling_module._fva_ranges_sha256(
            narrowed, prepared.lp.fingerprint
        ),
    )
    with pytest.raises(
        AnalysisError, match="numerically degenerate FVA interval.*PATH_1"
    ):
        sample_prepared_flux_states(prepared, 2, seed=65, fva=narrowed_result)


def test_true_fva_width_five_nanounits_survives_until_sampler_degeneracy_gate() -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("FIXED", (), 1.0, 1.0),
            FluxReaction("X", (), 0.0, 5e-9),
        ),
        LinearObjective("maximise", (ObjectiveTerm("FIXED", 1.0),)),
    )
    prepared = prepare_highs_flux_region(model, 1.0)
    fva = run_prepared_highs_vffva(prepared, workers=1)

    assert fva.ranges.loc["X", "minimum"] == pytest.approx(0.0, abs=1e-15)
    assert fva.ranges.loc["X", "maximum"] == pytest.approx(5e-9, abs=1e-15)
    assert (
        fva.ranges.loc["X", "maximum"] - fva.ranges.loc["X", "minimum"]
    ) == pytest.approx(5e-9, abs=1e-15)
    with pytest.raises(AnalysisError, match="numerically degenerate FVA interval.*X"):
        sample_prepared_flux_states(prepared, 2, seed=651, fva=fva)


def test_in_place_mutation_of_native_fva_ranges_is_detected_by_digest() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    supplied = run_prepared_highs_vffva(prepared, workers=1)
    fba_value = float(prepared.fba.fluxes["PATH_1"])
    supplied.ranges.loc["PATH_1", ["minimum", "maximum"]] = fba_value

    with pytest.raises(AnalysisError, match="FVA range digest.*mutated"):
        sample_prepared_flux_states(prepared, 2, seed=66, fva=supplied)


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        (
            pd.DataFrame(
                {
                    "minimum": (-1.0, 0.0, 0.0, 0.0),
                    "maximum": (10.0, 10.0, 10.0, 10.0),
                },
                index=("SOURCE", "PATH_1", "PATH_2", "BIOMASS"),
            ),
            "minimum.*lower bound.*SOURCE",
        ),
        (
            pd.DataFrame(
                {
                    "minimum": (8.0, 6.0, 0.0, 8.0),
                    "maximum": (10.0, 5.0, 10.0, 10.0),
                },
                index=("SOURCE", "PATH_1", "PATH_2", "BIOMASS"),
            ),
            "minimum exceeds maximum.*PATH_1",
        ),
    ],
)
def test_forged_supplied_fva_endpoints_are_rejected(
    ranges: pd.DataFrame, message: str
) -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    forged = FVAResult(
        ranges=ranges,
        fraction_of_optimum=prepared.retention.fraction_of_optimum,
        objective_value=prepared.retention.biological_optimum,
        objective_direction=prepared.retention.objective_direction,
        model_fingerprint=prepared.lp.fingerprint,
        ranges_sha256=sampling_module._fva_ranges_sha256(
            ranges, prepared.lp.fingerprint
        ),
    )

    with pytest.raises(AnalysisError, match=message):
        sample_prepared_flux_states(prepared, 2, seed=67, fva=forged)


def test_convenience_sampler_prepares_once_and_runs_one_fva(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_calls = 0
    fva_calls = 0
    original_prepare = sampling_module.prepare_highs_flux_region
    original_fva = sampling_module.run_prepared_highs_vffva

    def counted_prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    def counted_fva(*args: Any, **kwargs: Any) -> Any:
        nonlocal fva_calls
        fva_calls += 1
        return original_fva(*args, **kwargs)

    monkeypatch.setattr(sampling_module, "prepare_highs_flux_region", counted_prepare)
    monkeypatch.setattr(sampling_module, "run_prepared_highs_vffva", counted_fva)

    result = sample_highs_flux_states(
        _split_path_model(),
        2,
        fraction_of_optimum=0.8,
        fva_workers=1,
        seed=71,
        burn_in=2,
        thinning=1,
    )

    assert result.sample_count == 2
    assert prepare_calls == 1
    assert fva_calls == 1


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_sample_count_is_rejected(count: object) -> None:
    with pytest.raises(AnalysisError, match="count.*positive integer|sample count"):
        sample_highs_flux_states(
            _split_path_model(),
            count,  # type: ignore[arg-type]
            fraction_of_optimum=1.0,
            fva_workers=1,
            seed=1,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("seed", -1, "seed"),
        ("seed", 1.5, "seed"),
        ("seed", True, "seed"),
        ("burn_in", -1, "burn.in"),
        ("burn_in", 1.5, "burn.in"),
        ("burn_in", True, "burn.in"),
        ("thinning", 0, "thinning"),
        ("thinning", -1, "thinning"),
        ("thinning", 1.5, "thinning"),
        ("thinning", True, "thinning"),
        ("max_direction_attempts", 0, "direction.*attempt"),
        ("max_direction_attempts", -1, "direction.*attempt"),
        ("max_direction_attempts", 1.5, "direction.*attempt"),
        ("max_direction_attempts", True, "direction.*attempt"),
    ],
)
def test_invalid_sampler_parameter_is_rejected(
    keyword: str, value: object, message: str
) -> None:
    kwargs: dict[str, object] = {
        "seed": 1,
        "burn_in": 1,
        "thinning": 1,
        "max_direction_attempts": 10,
    }
    kwargs[keyword] = value
    prepared = prepare_highs_flux_region(_split_path_model(), 1.0)
    with pytest.raises(AnalysisError, match=message):
        sample_prepared_flux_states(prepared, 2, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01, math.nan, math.inf, True])
def test_invalid_fraction_is_rejected_before_sampling(fraction: object) -> None:
    with pytest.raises(AnalysisError, match="fraction_of_optimum"):
        sample_highs_flux_states(
            _split_path_model(),
            2,
            fraction_of_optimum=fraction,  # type: ignore[arg-type]
            fva_workers=1,
            seed=1,
        )


def test_malformed_prepared_region_is_rejected_at_sampler_entry() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    forged = replace(
        prepared,
        retention=replace(prepared.retention, bound=math.nan),
    )

    with pytest.raises(AnalysisError, match="retained-objective.*finite"):
        sample_prepared_flux_states(forged, 2, seed=1)


def test_stale_compiled_lp_fingerprint_is_rejected_at_sampler_entry() -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    mutated_lower_bounds = (-1.0,) + prepared.lp.lower_bounds[1:]
    forged = replace(
        prepared,
        lp=replace(prepared.lp, lower_bounds=mutated_lower_bounds),
    )

    with pytest.raises(AnalysisError, match="compiled LP fingerprint.*stale|inconsistent"):
        sample_prepared_flux_states(forged, 2, seed=1)


@pytest.mark.parametrize(
    ("case", "changes"),
    [
        ("row-count", {"row_starts": (0,)}),
        ("row-origin", {"row_starts": (1, 4, 6)}),
        ("column-index", {"column_indices": (4, 1, 1, 2, 2, 3)}),
        ("coefficient-count", {"coefficients": (1.0, -1.0, -1.0, 1.0, -1.0)}),
    ],
)
def test_malformed_compiled_csr_is_rejected_even_with_a_recomputed_fingerprint(
    case: str,
    changes: dict[str, tuple[object, ...]],
) -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    malformed_lp = replace(prepared.lp, **changes)
    malformed_lp = replace(
        malformed_lp,
        fingerprint=highs_module._compiled_lp_fingerprint(
            malformed_lp.reaction_ids,
            malformed_lp.balanced_metabolite_ids,
            malformed_lp.row_starts,
            malformed_lp.column_indices,
            malformed_lp.coefficients,
            malformed_lp.lower_bounds,
            malformed_lp.upper_bounds,
            malformed_lp.objective_coefficients,
            malformed_lp.objective_direction,
        ),
    )
    forged = replace(prepared, lp=malformed_lp)

    with pytest.raises(AnalysisError, match="prepared compiled LP|CSR|row"):
        sample_prepared_flux_states(forged, 2, seed=1)


def test_line_interval_rejects_an_unbounded_sampling_direction() -> None:
    with pytest.raises(AnalysisError, match="unbounded sampling direction"):
        sampling_module._line_interval(
            np.asarray(((1.0,),), dtype=float),
            np.asarray((1.0,), dtype=float),
            np.asarray((0.0,), dtype=float),
            np.asarray((1.0,), dtype=float),
        )


def test_line_interval_rejects_infeasible_current_point_on_parallel_row() -> None:
    with pytest.raises(
        AnalysisError, match="current point violates a retained inequality"
    ):
        sampling_module._line_interval(
            np.asarray(((0.0, 1.0), (1.0, 0.0), (-1.0, 0.0))),
            np.asarray((1.0, 1.0, 1.0)),
            np.asarray((0.0, 2.0)),
            np.asarray((1.0, 0.0)),
        )


def test_line_interval_exact_finite_self_loop_is_degenerate_not_an_error() -> None:
    interval = sampling_module._line_interval(
        np.asarray(((1.0,), (-1.0,)), dtype=float),
        np.asarray((0.0, 0.0), dtype=float),
        np.asarray((0.0,), dtype=float),
        np.asarray((1.0,), dtype=float),
    )

    assert interval is None


def test_tiny_projected_inequality_norm_is_scaled_without_underflow() -> None:
    geometry = sampling_module._ReducedFluxGeometry(
        particular=np.asarray((0.0,)),
        basis=np.asarray(((1.0,),)),
        inequality_rows=np.asarray(((1e-200,), (-1e-200,))),
        inequality_bounds=np.asarray((2e-200, 3e-200)),
        equality_rank=0,
        collapsed_reaction_ids=(),
        fixed_reaction_count=0,
        optimal_face=False,
        objective_scale=0.0,
        objective_constant=0.0,
        effective_objective_bound=0.0,
    )

    rows, bounds, norms = sampling_module._active_reduced_inequalities(geometry)

    np.testing.assert_allclose(rows, np.asarray(((1.0,), (-1.0,))))
    np.testing.assert_allclose(bounds, np.asarray((2.0, 3.0)))
    np.testing.assert_allclose(norms, np.asarray((1.0, 1.0)))


def test_line_interval_names_numerically_degenerate_chord() -> None:
    interval = sampling_module._line_interval(
        np.asarray(((1.0,), (-1.0,)), dtype=float),
        np.asarray((5e-13, 0.0), dtype=float),
        np.asarray((0.0,), dtype=float),
        np.asarray((1.0,), dtype=float),
    )

    assert interval is None


def test_line_interval_is_invariant_to_constraint_row_rescaling() -> None:
    point = np.asarray((0.0,), dtype=float)
    direction = np.asarray((1.0,), dtype=float)
    rows = np.asarray(((1.0,), (-1.0,)), dtype=float)
    bounds = np.asarray((1.0, 1.0), dtype=float)
    expected = sampling_module._line_interval(rows, bounds, point, direction)
    scaled = sampling_module._line_interval(
        rows * 1e-100, bounds * 1e-100, point, direction
    )
    assert scaled == pytest.approx(expected)


def test_nearly_parallel_active_face_still_limits_the_chord() -> None:
    interval = sampling_module._line_interval(
        np.asarray(((1e-15, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))),
        np.asarray((1e-16, 1.0, 1.0, 1.0)),
        np.asarray((0.0, 0.0)),
        np.asarray((1.0, 0.0)),
    )
    assert interval == pytest.approx((-1.0, 0.1))


def test_sampler_reports_exhausted_direction_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    supplied = run_prepared_highs_vffva(prepared, workers=1)

    class ZeroDirectionGenerator:
        def __init__(self, bit_generator: Any) -> None:
            self.bit_generator = bit_generator

        def standard_normal(self, dimension: int) -> np.ndarray:
            return np.zeros(dimension, dtype=float)

    monkeypatch.setattr(
        sampling_module.np.random, "Generator", ZeroDirectionGenerator
    )

    with pytest.raises(
        AnalysisError,
        match="could not generate a valid next state.*3 direction attempts.*accepted step 0",
    ):
        sample_prepared_flux_states(
            prepared,
            1,
            seed=73,
            fva=supplied,
            burn_in=0,
            thinning=1,
            max_direction_attempts=3,
        )


def test_sampler_counts_finite_degenerate_chords_as_self_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    supplied = run_prepared_highs_vffva(prepared, workers=1)
    monkeypatch.setattr(sampling_module, "_line_interval", lambda *args: None)

    result = sample_prepared_flux_states(
        prepared,
        3,
        seed=74,
        fva=supplied,
        burn_in=2,
        thinning=1,
        max_direction_attempts=3,
    )

    _assert_valid_samples(prepared, result)
    assert len({state.values for state in result.states}) == 1
    assert result.provenance.accepted_chain_steps == 5
    assert result.provenance.self_loop_steps == 5
    assert result.provenance.rejected_directions == 0


@pytest.mark.parametrize("case", ["nonfinite", "overflow"])
def test_chain_candidate_rejects_nonfinite_or_overflow_without_repair(
    case: str,
) -> None:
    prepared = prepare_highs_flux_region(_overflow_validation_model(), 1.0)
    values = (
        np.asarray((math.inf, 0.0, 1.0), dtype=float)
        if case == "nonfinite"
        else np.asarray((1.7e308, -1.7e308, 1.0), dtype=float)
    )

    with pytest.raises(AnalysisError) as caught:
        sampling_module._validate_chain_candidate(prepared, values, 7)

    message = str(caught.value).lower()
    assert "accepted step 7" in message
    assert "no repair or replacement" in message
    if case == "nonfinite":
        assert "non-finite ambient fluxes" in message
    else:
        assert "independent ambient-space validation" in message
        assert "conditioned_mass_residual=inf" in message


def test_chain_candidate_rejects_raw_mass_overflow_hidden_by_row_space() -> None:
    model = FluxModel(
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
    prepared = prepare_highs_flux_region(model, 1.0)

    with pytest.raises(AnalysisError) as caught:
        sampling_module._validate_chain_candidate(
            prepared, np.asarray((2.0, 2.0, 1.0)), 11
        )

    message = str(caught.value).lower()
    assert "accepted step 11" in message
    assert "raw_mass_residual=inf" in message
    assert "normalized_mass_residual=inf" in message
    assert "conditioned_mass_residual=0" in message
    assert "no repair or replacement" in message


def test_sampler_rejects_bad_burn_in_candidate_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_highs_flux_region(_split_path_model(), 0.8)
    supplied = run_prepared_highs_vffva(prepared, workers=1)
    calls = 0

    def outside_chord(*args: Any, **kwargs: Any) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return 100.0, 101.0

    monkeypatch.setattr(sampling_module, "_line_interval", outside_chord)
    with pytest.raises(
        AnalysisError, match="candidate violates.*no repair or replacement"
    ):
        sample_prepared_flux_states(
            prepared,
            1,
            seed=75,
            fva=supplied,
            burn_in=1,
            thinning=1,
            max_direction_attempts=3,
        )
    assert calls == 1


def test_sampler_aborts_on_declared_objective_discrepant_burn_in_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_highs_flux_region(
        _row_space_perturbed_objective_model(
            "maximise",
            1.0,
            row_scale=1e6,
        ),
        0.8,
    )
    supplied = run_prepared_highs_vffva(prepared, workers=1)
    original_validate_candidate = sampling_module._validate_chain_candidate
    injected_values = np.asarray(
        tuple(value for _, value in _near_mass_feasible_objective_state().values),
        dtype=float,
    )
    calls = 0

    def inject_discrepant_candidate(
        prepared_region: Any,
        candidate: np.ndarray,
        accepted_step: int,
    ) -> None:
        nonlocal calls
        calls += 1
        assert candidate.shape == injected_values.shape
        assert accepted_step == 0
        original_validate_candidate(
            prepared_region,
            injected_values,
            accepted_step,
        )

    monkeypatch.setattr(
        sampling_module,
        "_validate_chain_candidate",
        inject_discrepant_candidate,
    )
    with pytest.raises(
        AnalysisError,
        match=(
            "accepted step 0:.*declared_objective=.*stable_quotient=.*"
            "objective_discrepancy=.*direct_retention_violation=.*"
            "no repair or replacement"
        ),
    ):
        sample_prepared_flux_states(
            prepared,
            1,
            seed=751,
            fva=supplied,
            burn_in=1,
            thinning=1,
        )
    assert calls == 1


def test_nonfinite_bound_is_rejected_instead_of_truncating_unbounded_region() -> None:
    model = FluxModel(
        (),
        (FluxReaction("X", (), 0.0, math.inf),),
        LinearObjective("maximise", (ObjectiveTerm("X", 1.0),)),
    )
    with pytest.raises(AnalysisError, match="upper bound.*finite|invalid canonical"):
        sample_highs_flux_states(model, 2, seed=1, fva_workers=1)


def test_unrepresentable_finite_bound_span_fails_closed_without_numpy_warning() -> None:
    model = FluxModel(
        (),
        (
            FluxReaction("FIXED", (), 1.0, 1.0),
            FluxReaction("WIDE", (), -1e308, 1e308),
        ),
        LinearObjective("maximise", (ObjectiveTerm("FIXED", 1.0),)),
    )

    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(
            AnalysisError,
            match="floating-point range|unrepresentable|unbounded",
        ):
            sample_highs_flux_states(model, 1, seed=1, fva_workers=1)


def test_native_sampling_import_and_execution_firewall() -> None:
    script = r'''
import sys
from fluxemu.flux_analysis import sample_highs_flux_states
from fluxemu.model import FluxModel, FluxReaction, LinearObjective, ObjectiveTerm

model = FluxModel(
    (),
    (FluxReaction("FIXED", (), 2.0, 2.0), FluxReaction("SIGNED", (), -1.0, 1.0)),
    LinearObjective("maximise", (ObjectiveTerm("FIXED", 1.0),)),
)
result = sample_highs_flux_states(
    model, 2, fraction_of_optimum=1.0,
    seed=5, burn_in=2, thinning=1, max_direction_attempts=20,
)
assert len(result.states) == 2
for forbidden in ("cobra", "optlang", "mfapy", "scipy"):
    assert not any(name == forbidden or name.startswith(forbidden + ".") for name in sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr


def test_nontrivial_bundled_ecoli_model_returns_valid_complete_states() -> None:
    model = load_ecoli_core_flux_model("biomass")
    prepared = prepare_highs_flux_region(model, 1.0)
    result = sample_prepared_flux_states(
        prepared,
        2,
        seed=53,
        burn_in=8,
        thinning=2,
        max_direction_attempts=100,
    )

    _assert_valid_samples(prepared, result)
    assert len(prepared.lp.reaction_ids) == 95
    assert all(len(state.values) == 95 for state in result.states)
    assert tuple(result.to_frame().columns) == _reaction_order(model)
