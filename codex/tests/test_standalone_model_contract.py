"""Focused acceptance tests for the standalone canonical model contract."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fluxemu.model import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    CanonicalModelError,
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
    deterministic_serialise,
    experiment_fingerprint,
    model_fingerprint,
    validate_canonical_model,
    validate_stationary_experiment,
)


def canonical() -> CanonicalModel:
    flux = FluxModel(
        metabolites=(FluxMetabolite("B", False), FluxMetabolite("A", True)),
        reactions=(
            FluxReaction(
                "R",
                (StoichiometricTerm("B", -1.0), StoichiometricTerm("A", 1)),
                0.0,
                100,
            ),
        ),
        objective=LinearObjective("maximise", (ObjectiveTerm("R", 1.0),)),
    )
    isotope = IsotopeModel(
        metabolites=(
            IsotopeMetabolite("B", 2, True, False),
            IsotopeMetabolite("A", 2, True, True),
            IsotopeMetabolite("H2O", 0, False, False),
        ),
        reactions=(
            IsotopeReaction(
                reaction_id="R",
                direction="forward",
                isotope_enabled=True,
                substrates=(IsotopeParticipant("B", (2, 1)),),
                products=(IsotopeParticipant("A", (1, 2)),),
                mapping_branches=(
                    MappingBranch(
                        "declared",
                        1.0,
                        (
                            AtomTransition(AtomPosition("B", 2), AtomPosition("A", 1)),
                            AtomTransition(AtomPosition("B", 1), AtomPosition("A", 2)),
                        ),
                    ),
                ),
                directional_id="R:forward",
                symmetry_semantics="declared only; does not generate branches",
                provenance=(("source", "unit-test"),),
            ),
        ),
    )
    return CanonicalModel(flux, isotope)


def experiment() -> StationaryExperimentSemantics:
    return StationaryExperimentSemantics(
        tracers=(Tracer("B", (("#10", 0.25), ("#01", 0.75)), "no"),),
        targets=(Target("A-2,1", "A", (2, 1), "LC-MS", "C2H4O2", "yes"),),
    )


def invalid_model(*, flux=None, isotope=None) -> CanonicalModel:
    base = canonical()
    return replace(base, flux_model=flux or base.flux_model, isotope_model=isotope or base.isotope_model)


def assert_model_error(model: CanonicalModel) -> None:
    with pytest.raises(CanonicalModelError):
        validate_canonical_model(model)


def test_complete_mapping_and_stationary_experiment_are_accepted():
    validate_canonical_model(canonical())
    validate_stationary_experiment(canonical(), experiment())


def test_records_and_scientific_tuple_order_are_immutable_and_preserved():
    model = canonical()
    with pytest.raises(AttributeError):
        model.isotope_model.reactions[0].direction = "reverse"
    reaction = model.isotope_model.reactions[0]
    assert tuple(item.metabolite_id for item in reaction.substrates) == ("B",)
    assert reaction.substrates[0].atom_positions == (2, 1)
    assert tuple(item.source.position for item in reaction.mapping_branches[0].transitions) == (2, 1)
    assert experiment().targets[0].atom_positions == (2, 1)


def test_serialisation_is_deterministic_exact_and_preserves_order():
    first = deterministic_serialise(canonical())
    assert first == deterministic_serialise(canonical())
    assert '"$float":"0x1.0000000000000p+0"' in first
    assert first.index('"metabolite_id":"B"') < first.index('"metabolite_id":"A"')


def test_fingerprints_are_deterministic_and_separate_model_from_experiment():
    model = canonical()
    changed_experiment = replace(
        experiment(), targets=(replace(experiment().targets[0], correction="no"),)
    )
    assert model_fingerprint(model) == model_fingerprint(model)
    assert model_fingerprint(model) == model_fingerprint(model)
    assert experiment_fingerprint(model, experiment()) != experiment_fingerprint(model, changed_experiment)
    changed_model = replace(
        model,
        flux_model=replace(
            model.flux_model,
            reactions=(replace(model.flux_model.reactions[0], upper_bound=99),),
        ),
    )
    assert model_fingerprint(model) != model_fingerprint(changed_model)


@pytest.mark.parametrize("kind", ["flux_metabolite", "flux_reaction", "isotope_metabolite", "isotope_reaction"])
def test_duplicate_ids_rejected(kind):
    model = canonical()
    if kind == "flux_metabolite":
        flux = replace(model.flux_model, metabolites=model.flux_model.metabolites * 2)
        model = replace(model, flux_model=flux)
    elif kind == "flux_reaction":
        model = replace(model, flux_model=replace(model.flux_model, reactions=model.flux_model.reactions * 2))
    elif kind == "isotope_metabolite":
        duplicate = replace(model.isotope_model.metabolites[0], carbon_count=1)
        model = replace(model, isotope_model=replace(model.isotope_model, metabolites=model.isotope_model.metabolites + (duplicate,)))
    else:
        model = replace(model, isotope_model=replace(model.isotope_model, reactions=model.isotope_model.reactions * 2))
    assert_model_error(model)


@pytest.mark.parametrize(
    ("field", "value"),
    [("steady_state_balanced", 1), ("isotope_visible", 0), ("symmetry", "false"), ("isotope_enabled", None)],
)
def test_scientific_booleans_must_be_literal_bool(field, value):
    model = canonical()
    if field == "steady_state_balanced":
        metabolite = replace(model.flux_model.metabolites[0], **{field: value})
        model = replace(model, flux_model=replace(model.flux_model, metabolites=(metabolite,) + model.flux_model.metabolites[1:]))
    elif field in {"isotope_visible", "symmetry"}:
        metabolite = replace(model.isotope_model.metabolites[0], **{field: value})
        model = replace(model, isotope_model=replace(model.isotope_model, metabolites=(metabolite,) + model.isotope_model.metabolites[1:]))
    else:
        reaction = replace(model.isotope_model.reactions[0], **{field: value})
        model = replace(model, isotope_model=replace(model.isotope_model, reactions=(reaction,)))
    assert_model_error(model)


@pytest.mark.parametrize("carbon_count", [True, 1.0, -1, "2", None])
def test_carbon_count_integer_semantics(carbon_count):
    model = canonical()
    metabolite = replace(model.isotope_model.metabolites[0], carbon_count=carbon_count)
    assert_model_error(replace(model, isotope_model=replace(model.isotope_model, metabolites=(metabolite,) + model.isotope_model.metabolites[1:])))


def test_visible_zero_carbon_rejected_and_invisible_zero_carbon_accepted():
    model = canonical()
    assert model.isotope_model.metabolites[-1].carbon_count == 0
    validate_canonical_model(model)
    bad = replace(model.isotope_model.metabolites[0], carbon_count=0)
    assert_model_error(replace(model, isotope_model=replace(model.isotope_model, metabolites=(bad,) + model.isotope_model.metabolites[1:])))


@pytest.mark.parametrize("direction", ["Forward", "backward", "max", None])
def test_isotope_direction_is_exact(direction):
    model = canonical()
    reaction = replace(model.isotope_model.reactions[0], direction=direction)
    assert_model_error(replace(model, isotope_model=replace(model.isotope_model, reactions=(reaction,))))


@pytest.mark.parametrize("direction", ["maximize", "max", "MINIMISE", None])
def test_objective_direction_is_exact(direction):
    model = canonical()
    objective = replace(model.flux_model.objective, direction=direction)
    assert_model_error(replace(model, flux_model=replace(model.flux_model, objective=objective)))


@pytest.mark.parametrize("coefficient", [True, "1", None, object(), math.nan, math.inf, -math.inf])
def test_objective_coefficients_are_finite_numbers_not_bool(coefficient):
    model = canonical()
    objective = replace(model.flux_model.objective, terms=(ObjectiveTerm("R", coefficient),))
    assert_model_error(replace(model, flux_model=replace(model.flux_model, objective=objective)))


@pytest.mark.parametrize("weight", [True, 0, -0.1, "1", None, math.nan, math.inf])
def test_mapping_branch_weights_are_positive_finite_numbers(weight):
    model = canonical()
    reaction = model.isotope_model.reactions[0]
    branch = replace(reaction.mapping_branches[0], weight=weight)
    reaction = replace(reaction, mapping_branches=(branch,))
    assert_model_error(replace(model, isotope_model=replace(model.isotope_model, reactions=(reaction,))))


def test_mapping_branch_mixture_must_normalize():
    model = canonical()
    reaction = model.isotope_model.reactions[0]
    branches = (replace(reaction.mapping_branches[0], branch_id="one", weight=0.4), replace(reaction.mapping_branches[0], branch_id="two", weight=0.5))
    assert_model_error(replace(model, isotope_model=replace(model.isotope_model, reactions=(replace(reaction, mapping_branches=branches),))))


@pytest.mark.parametrize(
    "participant",
    [IsotopeParticipant("missing", (1,)), IsotopeParticipant("H2O", (1,)), IsotopeParticipant("B", ()), IsotopeParticipant("B", (True,)), IsotopeParticipant("B", (1, 1)), IsotopeParticipant("B", (3,))],
)
def test_participant_reference_visibility_and_atom_rules(participant):
    model = canonical()
    reaction = replace(model.isotope_model.reactions[0], substrates=(participant,))
    assert_model_error(replace(model, isotope_model=replace(model.isotope_model, reactions=(reaction,))))


def mapped_model(transitions) -> CanonicalModel:
    model = canonical()
    reaction = model.isotope_model.reactions[0]
    branch = replace(reaction.mapping_branches[0], transitions=tuple(transitions))
    return replace(model, isotope_model=replace(model.isotope_model, reactions=(replace(reaction, mapping_branches=(branch,)),)))


S1 = AtomPosition("B", 1)
S2 = AtomPosition("B", 2)
D1 = AtomPosition("A", 1)
D2 = AtomPosition("A", 2)


@pytest.mark.parametrize(
    "transitions",
    [
        (AtomTransition(S2, D1),),
        (AtomTransition(S2, D1), AtomTransition(S1, D1)),
        (AtomTransition(S2, D1), AtomTransition(S2, D2)),
        (AtomTransition(S2, D1), AtomTransition(AtomPosition("X", 1), D2)),
        (AtomTransition(S2, D1), AtomTransition(S1, AtomPosition("X", 1))),
        (AtomTransition(S2, D1), AtomTransition(S1, D2), AtomTransition(AtomPosition("B", 3), AtomPosition("A", 3))),
        (),
        ("malformed",),
    ],
)
def test_incomplete_extra_duplicate_undeclared_or_malformed_maps_rejected(transitions):
    assert_model_error(mapped_model(transitions))


def experiment_with_tracer(**changes):
    tracer = replace(experiment().tracers[0], **changes)
    return replace(experiment(), tracers=(tracer,))


@pytest.mark.parametrize(
    "changes",
    [
        {"metabolite_id": "missing"},
        {"metabolite_id": "H2O"},
        {"isotopomers": ()},
        {"isotopomers": (("10", 1.0),)},
        {"isotopomers": (("#1", 1.0),)},
        {"isotopomers": (("#1x", 1.0),)},
        {"isotopomers": (("#10", 0.5), ("#10", 0.5))},
        {"isotopomers": (("#10", True),)},
        {"isotopomers": (("#10", math.nan),)},
        {"isotopomers": (("#10", 1.1), ("#01", -0.1))},
        {"isotopomers": (("#10", 0.4), ("#01", 0.5))},
        {"correction": "true"},
    ],
)
def test_tracer_validation(changes):
    with pytest.raises(CanonicalModelError):
        validate_stationary_experiment(canonical(), experiment_with_tracer(**changes))


def test_tracer_order_preserved():
    tracer = experiment().tracers[0]
    validate_stationary_experiment(canonical(), experiment())
    assert tracer.isotopomers == (("#10", 0.25), ("#01", 0.75))


def experiment_with_target(**changes):
    target = replace(experiment().targets[0], **changes)
    return replace(experiment(), targets=(target,))


@pytest.mark.parametrize(
    "changes",
    [
        {"target_id": ""},
        {"metabolite_id": "missing"},
        {"metabolite_id": "H2O"},
        {"atom_positions": ()},
        {"atom_positions": (True,)},
        {"atom_positions": (1.0,)},
        {"atom_positions": (1, 1)},
        {"atom_positions": (3,)},
        {"analytical_method": ""},
        {"formula": ""},
        {"correction": "true"},
    ],
)
def test_target_validation(changes):
    with pytest.raises(CanonicalModelError):
        validate_stationary_experiment(canonical(), experiment_with_target(**changes))


def test_experiment_declared_tracer_and_target_order_preserved():
    exp = experiment()
    second_tracer = replace(exp.tracers[0], metabolite_id="A")
    second_target = replace(exp.targets[0], target_id="second")
    ordered = replace(exp, tracers=(second_tracer, exp.tracers[0]), targets=(second_target, exp.targets[0]))
    validate_stationary_experiment(canonical(), ordered)
    assert ordered.tracers[0] is second_tracer
    assert ordered.targets[0] is second_target


def test_fluxemu_model_import_is_engine_independent():
    import fluxemu.model.schema as model_schema

    source = Path(model_schema.__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source)
    command = "import sys; import fluxemu.model; assert not {'cobra','mfapy','matplotlib'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", command], env=environment, check=True)
