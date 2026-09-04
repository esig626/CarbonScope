"""Independent black-box acceptance for the complete native Stage 1 ensemble."""

from __future__ import annotations

from dataclasses import fields
import math
import subprocess
import sys

import pytest

import fluxemu
from fluxemu.analysis import (
    NativeStationaryEnsembleResult,
    run_native_stationary_ensemble,
)
from fluxemu.execution import CanonicalFluxState
from fluxemu.flux_analysis import validate_flux_states
from fluxemu.model import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    FluxMetabolite,
    FluxModel,
    FluxReaction,
    IsotopeMetabolite,
    IsotopeModel,
    IsotopeParticipant,
    IsotopeReaction,
    LinearObjective,
    MappingBranch,
    ObjectiveTerm,
    StationaryExperimentSemantics,
    StoichiometricTerm,
    Target,
    Tracer,
)
from fluxemu.real_model import (
    TARGET_COVERAGE,
    build_r1_acceptance_experiment,
    load_ecoli_core_stage_b2_model,
)


FLUX_TOLERANCE = 1e-7
MID_TOLERANCE = 1e-10


def _mapped_reaction(
    reaction_id: str, source: str, product: str
) -> IsotopeReaction:
    transition = AtomTransition(
        AtomPosition(source, 1), AtomPosition(product, 1)
    )
    return IsotopeReaction(
        reaction_id,
        "forward",
        True,
        (IsotopeParticipant(source, (1,)),),
        (IsotopeParticipant(product, (1,)),),
        (MappingBranch("declared", 1.0, (transition,)),),
    )


def _alternate_label_science(
    *,
    objective_direction: str = "maximise",
    objective_coefficient: float = 1.0,
) -> tuple[CanonicalModel, StationaryExperimentSemantics]:
    """Return two alternate optimal inputs carrying distinguishable labels."""

    metabolites = (
        FluxMetabolite("S0", False),
        FluxMetabolite("S1", False),
        FluxMetabolite("I", True),
        FluxMetabolite("O", False),
    )
    # Non-lexical order is deliberate and must survive every Stage 1 layer.
    reactions = (
        FluxReaction(
            "Z_IN",
            (
                StoichiometricTerm("S0", -1.0),
                StoichiometricTerm("I", 1.0),
            ),
            0.0,
            10.0,
        ),
        FluxReaction(
            "A_IN",
            (
                StoichiometricTerm("S1", -1.0),
                StoichiometricTerm("I", 1.0),
            ),
            0.0,
            10.0,
        ),
        FluxReaction(
            "M_OUT",
            (
                StoichiometricTerm("I", -1.0),
                StoichiometricTerm("O", 1.0),
            ),
            0.0,
            10.0,
        ),
    )
    model = CanonicalModel(
        FluxModel(
            metabolites,
            reactions,
            LinearObjective(
                objective_direction,
                (ObjectiveTerm("M_OUT", objective_coefficient),),
            ),
        ),
        IsotopeModel(
            tuple(
                IsotopeMetabolite(item.metabolite_id, 1, True, False)
                for item in metabolites
            ),
            (
                _mapped_reaction("Z_IN", "S0", "I"),
                _mapped_reaction("A_IN", "S1", "I"),
                _mapped_reaction("M_OUT", "I", "O"),
            ),
        ),
    )
    experiment = StationaryExperimentSemantics(
        (
            Tracer("S0", (("#0", 1.0),), "no"),
            Tracer("S1", (("#1", 1.0),), "no"),
        ),
        (Target("O-mid", "O", (1,), "intermediate", "C1", "no"),),
    )
    return model, experiment


def _run_alternate_ensemble(
    *,
    count: int = 6,
    seed: int = 1729,
    fraction: float = 1.0,
    objective_direction: str = "maximise",
    objective_coefficient: float = 1.0,
) -> tuple[CanonicalModel, NativeStationaryEnsembleResult]:
    model, experiment = _alternate_label_science(
        objective_direction=objective_direction,
        objective_coefficient=objective_coefficient,
    )
    result = fluxemu.run_native_stationary_ensemble(
        model,
        experiment,
        sample_count=count,
        seed=seed,
        fraction_of_optimum=fraction,
        burn_in=12,
        thinning=2,
        fva_workers=1,
        max_direction_attempts=100,
    )
    return model, result


def _independent_flux_validation(
    model: CanonicalModel, result: NativeStationaryEnsembleResult
):
    provenance = result.flux_sampling.provenance
    return validate_flux_states(
        model.flux_model,
        result.flux_sampling.states,
        retained_objective_bound=provenance.retained_objective_bound,
        objective_direction=provenance.objective_direction,
        optimal_face_value=(
            provenance.biological_optimum if provenance.objective_face else None
        ),
        bounds_tolerance=provenance.bounds_tolerance,
        mass_balance_tolerance=provenance.mass_balance_tolerance,
        objective_tolerance=provenance.objective_tolerance,
    )


def _assert_valid_mid_batch(result: NativeStationaryEnsembleResult) -> None:
    forward = result.mid_ensemble.forward
    assert forward.max_normalization_error <= forward.mid_tolerance
    for prediction in forward.predictions:
        assert prediction.fractions
        assert all(math.isfinite(value) for value in prediction.fractions)
        assert all(value >= 0.0 for value in prediction.fractions)
        assert sum(prediction.fractions) == pytest.approx(1.0, abs=MID_TOLERANCE)
    assert tuple(
        (value.sample_id, value.target_id, value.isotopologue_index)
        for value in forward.values
    ) == tuple(
        (prediction.sample_id, prediction.target_id, index)
        for prediction in forward.predictions
        for index in range(len(prediction.fractions))
    )
    assert tuple(value.predicted_fraction for value in forward.values) == tuple(
        fraction
        for prediction in forward.predictions
        for fraction in prediction.fractions
    )


def test_public_stage1_result_separates_products_and_preserves_exact_identity() -> None:
    model, result = _run_alternate_ensemble()

    assert fluxemu.NativeStationaryEnsembleResult is NativeStationaryEnsembleResult
    assert fluxemu.run_native_stationary_ensemble is run_native_stationary_ensemble
    assert isinstance(result, NativeStationaryEnsembleResult)
    assert tuple(field.name for field in fields(result)) == (
        "fba",
        "fva",
        "flux_sampling",
        "mid_ensemble",
    )

    reaction_order = ("Z_IN", "A_IN", "M_OUT")
    sample_ids = tuple(f"native-sample-{index:06d}" for index in range(6))
    assert result.fba.status == "optimal"
    assert result.fba.objective_value == pytest.approx(10.0)
    assert tuple(result.fba.fluxes.index) == reaction_order
    assert tuple(result.fva.ranges.index) == reaction_order
    assert tuple(result.fva.ranges.columns) == ("minimum", "maximum")
    assert result.fva.fraction_of_optimum == 1.0
    assert result.fva.objective_value == pytest.approx(10.0)
    assert result.fva.objective_direction == "max"

    sampling = result.flux_sampling
    assert sampling.sample_count == 6
    assert sampling.validation.valid
    assert _independent_flux_validation(model, result).valid
    assert tuple(state.sample_id for state in sampling.states) == sample_ids
    assert all(
        tuple(reaction_id for reaction_id, _ in state.values) == reaction_order
        for state in sampling.states
    )
    assert all(
        dict(state.values)["Z_IN"] + dict(state.values)["A_IN"]
        == pytest.approx(dict(state.values)["M_OUT"], abs=FLUX_TOLERANCE)
        for state in sampling.states
    )
    assert all(
        dict(state.values)["M_OUT"] == pytest.approx(10.0, abs=FLUX_TOLERANCE)
        for state in sampling.states
    )
    assert len({state.values for state in sampling.states}) > 1

    predictions = result.mid_ensemble.forward.predictions
    assert tuple((item.sample_id, item.target_id) for item in predictions) == tuple(
        (sample_id, "O-mid") for sample_id in sample_ids
    )
    assert len(predictions) == sampling.sample_count
    assert len(result.mid_ensemble.forward.values) == 2 * sampling.sample_count
    assert tuple(
        diagnostic.sample_id for diagnostic in result.mid_ensemble.layer_diagnostics
    ) == sample_ids
    _assert_valid_mid_batch(result)

    labeled_fractions = []
    for state, prediction in zip(sampling.states, predictions, strict=True):
        flux = dict(state.values)
        expected_labeled = flux["A_IN"] / flux["M_OUT"]
        assert prediction.fractions == pytest.approx(
            (1.0 - expected_labeled, expected_labeled), abs=MID_TOLERANCE
        )
        labeled_fractions.append(prediction.fractions[1])
    assert len(set(labeled_fractions)) > 1


def test_fixed_seed_replays_flux_states_provenance_and_conditional_mids() -> None:
    _, first = _run_alternate_ensemble(seed=8675309)
    _, second = _run_alternate_ensemble(seed=8675309)

    assert first.fba.objective_value == second.fba.objective_value
    assert first.fba.status == second.fba.status
    assert first.fba.objective_direction == second.fba.objective_direction
    assert first.fba.fluxes.equals(second.fba.fluxes)
    assert first.fba.diagnostics == second.fba.diagnostics
    assert first.fva.ranges.equals(second.fva.ranges)
    assert first.fva.fraction_of_optimum == second.fva.fraction_of_optimum
    assert first.fva.objective_value == second.fva.objective_value
    assert first.fva.objective_direction == second.fva.objective_direction
    assert first.flux_sampling == second.flux_sampling
    assert first.mid_ensemble == second.mid_ensemble


def test_fva_endpoint_columns_are_not_states_or_the_source_of_mids() -> None:
    model, result = _run_alternate_ensemble()
    provenance = result.flux_sampling.provenance
    sampled_values = {state.values for state in result.flux_sampling.states}

    for column in ("minimum", "maximum"):
        endpoint = CanonicalFluxState(
            f"fva-{column}",
            tuple(
                (reaction_id, float(result.fva.ranges.loc[reaction_id, column]))
                for reaction_id in provenance.reaction_ids
            ),
        )
        report = validate_flux_states(
            model.flux_model,
            (endpoint,),
            retained_objective_bound=provenance.retained_objective_bound,
            objective_direction=provenance.objective_direction,
            optimal_face_value=provenance.biological_optimum,
        )
        assert not report.valid
        assert not report.mass_balance_valid
        assert endpoint.values not in sampled_values

    # Every emitted MID is instead conditional on its same-ID complete sample.
    by_id = {state.sample_id: dict(state.values) for state in result.flux_sampling.states}
    for prediction in result.mid_ensemble.forward.predictions:
        flux = by_id[prediction.sample_id]
        assert prediction.fractions[1] == pytest.approx(
            flux["A_IN"] / flux["M_OUT"], abs=MID_TOLERANCE
        )


@pytest.mark.parametrize(
    (
        "objective_direction",
        "objective_coefficient",
        "fraction",
        "expected_direction",
        "expected_optimum",
        "expected_bound",
        "expected_sense",
        "objective_face",
    ),
    (
        ("maximise", 1.0, 1.0, "max", 10.0, 10.0, ">=", True),
        ("maximise", 1.0, 0.8, "max", 10.0, 8.0, ">=", False),
        ("minimise", -1.0, 1.0, "min", -10.0, -10.0, "<=", True),
        ("minimise", -1.0, 0.8, "min", -10.0, -8.0, "<=", False),
    ),
)
def test_composed_pipeline_preserves_maximum_and_minimum_retention_semantics(
    objective_direction: str,
    objective_coefficient: float,
    fraction: float,
    expected_direction: str,
    expected_optimum: float,
    expected_bound: float,
    expected_sense: str,
    objective_face: bool,
) -> None:
    model, result = _run_alternate_ensemble(
        count=3,
        seed=91,
        fraction=fraction,
        objective_direction=objective_direction,
        objective_coefficient=objective_coefficient,
    )
    provenance = result.flux_sampling.provenance

    assert result.fba.objective_direction == expected_direction
    assert result.fba.objective_value == pytest.approx(expected_optimum)
    assert result.fva.objective_direction == expected_direction
    assert result.fva.fraction_of_optimum == fraction
    assert provenance.fraction_of_optimum == fraction
    assert provenance.biological_optimum == pytest.approx(expected_optimum)
    assert provenance.retained_objective_bound == pytest.approx(expected_bound)
    assert provenance.retained_objective_sense == expected_sense
    assert provenance.objective_direction == expected_direction
    assert provenance.objective_face is objective_face
    assert result.flux_sampling.validation.valid
    assert _independent_flux_validation(model, result).valid
    _assert_valid_mid_batch(result)

    for state in result.flux_sampling.states:
        objective_value = objective_coefficient * dict(state.values)["M_OUT"]
        if expected_direction == "max":
            assert objective_value >= expected_bound - FLUX_TOLERANCE
        else:
            assert objective_value <= expected_bound + FLUX_TOLERANCE
        if objective_face:
            assert objective_value == pytest.approx(
                expected_optimum, abs=FLUX_TOLERANCE
            )


def test_ecoli_stage_b2_produces_ordered_valid_twelve_mid_ensemble() -> None:
    model = load_ecoli_core_stage_b2_model("biomass")
    experiment = build_r1_acceptance_experiment()
    result = run_native_stationary_ensemble(
        model,
        experiment,
        sample_count=2,
        seed=20260904,
        fraction_of_optimum=1.0,
        burn_in=20,
        thinning=3,
        fva_workers=1,
        max_direction_attempts=200,
    )

    reaction_order = tuple(
        reaction.reaction_id for reaction in model.flux_model.reactions
    )
    sample_ids = ("native-sample-000000", "native-sample-000001")
    target_order = tuple(item.display_name for item in TARGET_COVERAGE)
    assert len(reaction_order) == 95
    assert result.fba.status == "optimal"
    assert result.fba.objective_value == pytest.approx(
        0.8739215069684307, abs=FLUX_TOLERANCE
    )
    assert tuple(result.fba.fluxes.index) == reaction_order
    assert result.fva.ranges.shape == (95, 2)
    assert tuple(result.fva.ranges.index) == reaction_order
    assert result.flux_sampling.sample_count == 2
    assert result.flux_sampling.validation.valid
    assert _independent_flux_validation(model, result).valid
    assert tuple(
        state.sample_id for state in result.flux_sampling.states
    ) == sample_ids
    assert all(
        tuple(reaction_id for reaction_id, _ in state.values) == reaction_order
        for state in result.flux_sampling.states
    )
    assert len({state.values for state in result.flux_sampling.states}) == 2

    predictions = result.mid_ensemble.forward.predictions
    assert len(predictions) == 2 * len(target_order) == 24
    assert tuple((item.sample_id, item.target_id) for item in predictions) == tuple(
        (sample_id, target_id)
        for sample_id in sample_ids
        for target_id in target_order
    )
    assert tuple(len(item.fractions) for item in predictions) == tuple(
        item.requested_carbon_count + 1
        for _sample_id in sample_ids
        for item in TARGET_COVERAGE
    )
    assert len(result.mid_ensemble.forward.values) == 116
    assert tuple(
        dict.fromkeys(
            diagnostic.sample_id
            for diagnostic in result.mid_ensemble.layer_diagnostics
        )
    ) == sample_ids
    _assert_valid_mid_batch(result)


def test_root_stage1_ensemble_import_and_execution_need_no_optional_stacks() -> None:
    script = r'''
import sys
sys.modules["cobra"] = None
sys.modules["optlang"] = None
sys.modules["mfapy"] = None

import fluxemu
assert sys.modules["cobra"] is None
assert sys.modules["optlang"] is None
assert sys.modules["mfapy"] is None

from fluxemu.model import (
    AtomPosition, AtomTransition, CanonicalModel, FluxMetabolite, FluxModel,
    FluxReaction, IsotopeMetabolite, IsotopeModel, IsotopeParticipant,
    IsotopeReaction, LinearObjective, MappingBranch, ObjectiveTerm,
    StationaryExperimentSemantics, StoichiometricTerm, Target, Tracer,
)

transition = AtomTransition(AtomPosition("S", 1), AtomPosition("P", 1))
model = CanonicalModel(
    FluxModel(
        (FluxMetabolite("S", False), FluxMetabolite("P", False)),
        (FluxReaction(
            "R", (StoichiometricTerm("S", -1), StoichiometricTerm("P", 1)),
            1, 1,
        ),),
        LinearObjective("maximise", (ObjectiveTerm("R", 1),)),
    ),
    IsotopeModel(
        (IsotopeMetabolite("S", 1, True, False),
         IsotopeMetabolite("P", 1, True, False)),
        (IsotopeReaction(
            "R", "forward", True, (IsotopeParticipant("S", (1,)),),
            (IsotopeParticipant("P", (1,)),),
            (MappingBranch("declared", 1, (transition,)),),
        ),),
    ),
)
experiment = StationaryExperimentSemantics(
    (Tracer("S", (("#1", 1),), "no"),),
    (Target("P-mid", "P", (1,), "intermediate", "C1", "no"),),
)
result = fluxemu.run_native_stationary_ensemble(
    model, experiment, sample_count=2, seed=7, burn_in=0, thinning=1,
    fva_workers=1,
)
assert result.flux_sampling.validation.valid
assert tuple(state.sample_id for state in result.flux_sampling.states) == (
    "native-sample-000000", "native-sample-000001",
)
assert tuple(
    (item.sample_id, item.target_id, item.fractions)
    for item in result.mid_ensemble.forward.predictions
) == (
    ("native-sample-000000", "P-mid", (0.0, 1.0)),
    ("native-sample-000001", "P-mid", (0.0, 1.0)),
)
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
