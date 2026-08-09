"""Focused acceptance tests for authoritative COBRA compatibility projection."""

from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction
from cobra.io import read_sbml_model

from fluxemu.carbon_transitions import load_default_library
from fluxemu.configuration import load_experiment
from fluxemu.compat import (
    AuthoritativeTransitionAssignment as Assignment,
    ProjectionError,
    project_cobra_model,
    project_stationary_experiment,
    resolve_authoritative_transitions,
)
from fluxemu.isotope_metadata import (
    AtomMappedParticipant,
    MetaboliteIsotopeMetadata,
    ReactionIsotopeMetadata,
    set_metabolite_metadata,
    set_reaction_metadata,
)
from fluxemu.model import (
    experiment_fingerprint,
    model_fingerprint,
    validate_canonical_model,
    validate_stationary_experiment,
)


LIBRARY = load_default_library()
PRIMARY_ID = "glycolysis.pyruvate_kinase"
WEIGHTED_ID = "antoniewicz.table5.v5.succinate_to_fumarate"


def cobra_reaction(substrate="phosphoenolpyruvate", product="pyruvate", reaction_id="R"):
    model = Model("projection")
    source = Metabolite(substrate, compartment="c")
    destination = Metabolite(product, compartment="c")
    reaction = Reaction(reaction_id, lower_bound=0, upper_bound=37)
    reaction.add_metabolites({source: -1, destination: 1})
    model.add_reactions([reaction])
    return model, reaction, source, destination


def assignment(reaction_id="R", transition_id=PRIMARY_ID, direction="forward"):
    return Assignment(reaction_id, transition_id, direction)


def legacy_metadata(model, reaction, source, destination, *, symmetry=False, labels=("b", "a")):
    for metabolite, is_source in ((source, True), (destination, False)):
        set_metabolite_metadata(
            metabolite,
            MetaboliteIsotopeMetadata(metabolite.id, 2, is_source, not is_source, symmetry, True),
        )
    set_reaction_metadata(
        reaction,
        ReactionIsotopeMetadata(
            reaction.id, f"{reaction.id}:forward", "forward", True,
            (AtomMappedParticipant(source.id, labels),),
            (AtomMappedParticipant(destination.id, tuple(reversed(labels))),),
        ),
    )


def test_flux_order_bounds_balance_and_objective_order_are_preserved():
    model = Model("ordered")
    z = Metabolite("z", compartment="c")
    a = Metabolite("a", compartment="c")
    second = Reaction("second", lower_bound=-3, upper_bound=9)
    second.add_metabolites({z: -2, a: 2})
    first = Reaction("first", lower_bound=0, upper_bound=4)
    first.add_metabolites({a: -1, z: 1})
    model.add_reactions([second, first])
    model.objective = {second: 3, first: -2}
    model.objective_direction = "min"
    set_metabolite_metadata(z, MetaboliteIsotopeMetadata("z", 1, True, False, False, True))

    projected = project_cobra_model(model)
    assert tuple(x.metabolite_id for x in projected.flux_model.metabolites) == ("z", "a")
    assert tuple(x.reaction_id for x in projected.flux_model.reactions) == ("second", "first")
    assert tuple(x.metabolite_id for x in projected.flux_model.reactions[0].stoichiometric_terms) == ("z", "a")
    assert (projected.flux_model.reactions[0].lower_bound, projected.flux_model.reactions[0].upper_bound) == (-3, 9)
    assert tuple(x.steady_state_balanced for x in projected.flux_model.metabolites) == (False, True)
    assert projected.flux_model.objective.direction == "minimise"
    assert tuple(x.reaction_id for x in projected.flux_model.objective.terms) == ("second", "first")


def test_authoritative_primary_participants_atoms_provenance_and_validation():
    model, _, _, _ = cobra_reaction()
    projected = project_cobra_model(model, [assignment()])
    isotope = projected.isotope_model.reactions[0]
    assert isotope.substrates[0].metabolite_id == "phosphoenolpyruvate"
    assert isotope.substrates[0].atom_positions == (1, 2, 3)
    assert isotope.products[0].atom_positions == (1, 2, 3)
    assert [(x.branch_id, x.weight) for x in isotope.mapping_branches] == [("primary", 1.0)]
    assert isotope.directional_id == f"{PRIMARY_ID}:forward"
    assert dict(isotope.provenance)["source_identifier"]
    validate_canonical_model(projected)


def test_weighted_branch_and_transition_order_are_copied_exactly():
    model, _, _, _ = cobra_reaction("succinate", "fumarate")
    projected = project_cobra_model(model, [assignment(transition_id=WEIGHTED_ID)])
    actual = projected.isotope_model.reactions[0].mapping_branches
    expected = LIBRARY.by_id[WEIGHTED_ID].forward_branches
    assert tuple((x.branch_id, x.weight) for x in actual) == tuple((x.branch_id, x.weight) for x in expected)
    assert tuple((x.source.position, x.destination.position) for x in actual[1].transitions) == tuple(
        (x.source.position, x.destination.position) for x in expected[1].atom_map
    )


def test_authoritative_weighted_map_is_independent_of_lossy_legacy_primary_map():
    model, reaction, source, destination = cobra_reaction("succinate", "fumarate")
    legacy_metadata(model, reaction, source, destination, symmetry=True, labels=("a", "b"))
    first = project_cobra_model(model, [assignment(transition_id=WEIGHTED_ID)])
    legacy_metadata(model, reaction, source, destination, symmetry=True, labels=("x", "y"))
    second = project_cobra_model(model, [assignment(transition_id=WEIGHTED_ID)])
    assert first.isotope_model.reactions[0].mapping_branches == second.isotope_model.reactions[0].mapping_branches


def test_complete_explicit_non_symmetric_legacy_map_is_projected_in_declared_order():
    model, reaction, source, destination = cobra_reaction("source", "product")
    legacy_metadata(model, reaction, source, destination)
    isotope = project_cobra_model(model).isotope_model.reactions[0]
    assert isotope.substrates[0].atom_positions == (1, 2)
    assert tuple((x.source.position, x.destination.position) for x in isotope.mapping_branches[0].transitions) == ((1, 2), (2, 1))
    assert isotope.mapping_branches[0].weight == 1.0


def test_symmetric_legacy_primary_is_rejected_and_boolean_cannot_supply_weights():
    model, reaction, source, destination = cobra_reaction("source", "product")
    legacy_metadata(model, reaction, source, destination, symmetry=True)
    with pytest.raises(ProjectionError, match="R.*authoritative weighted transition information is required"):
        project_cobra_model(model)


def test_missing_and_incomplete_legacy_map_are_rejected():
    model, reaction, source, destination = cobra_reaction("source", "product")
    legacy_metadata(model, reaction, source, destination)
    broken = ReactionIsotopeMetadata(
        "R", "R:forward", "forward", True,
        (AtomMappedParticipant("source", ("a", "b")),),
        (AtomMappedParticipant("product", ("a", "c")),),
    )
    set_reaction_metadata(reaction, broken)
    with pytest.raises(ProjectionError, match="incomplete destination coverage"):
        project_cobra_model(model)


def test_missing_legacy_reaction_map_and_boolean_alone_do_not_create_branches():
    model, _, source, destination = cobra_reaction("source", "product")
    for metabolite in (source, destination):
        set_metabolite_metadata(
            metabolite, MetaboliteIsotopeMetadata(metabolite.id, 2, False, False, True, True)
        )
    with pytest.raises(ProjectionError, match="missing an explicit legacy map"):
        project_cobra_model(model)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([assignment(), assignment()], "duplicate"),
        ([assignment(transition_id="does.not.exist")], "unknown authoritative transition"),
        ([assignment(reaction_id="missing")], "unknown reaction"),
    ],
)
def test_assignment_identity_failures(items, message):
    model, _, _, _ = cobra_reaction()
    with pytest.raises(ProjectionError, match=message):
        resolve_authoritative_transitions(model, items)


def test_complete_manifest_rejects_missing_and_extra_assignments():
    model, reaction, source, destination = cobra_reaction()
    legacy_metadata(model, reaction, source, destination)
    with pytest.raises(ProjectionError, match="missing assignment"):
        resolve_authoritative_transitions(model, [], require_complete=True)
    blank, _, _, _ = cobra_reaction()
    with pytest.raises(ProjectionError, match="extra assignment"):
        resolve_authoritative_transitions(blank, [assignment()], require_complete=True)


def test_explicit_reverse_direction_is_verified_not_guessed():
    model, _, _, _ = cobra_reaction()
    with pytest.raises(ProjectionError, match="reverse direction"):
        resolve_authoritative_transitions(model, [assignment(direction="reverse")])


def test_assignment_order_is_retained_in_resolved_mapping():
    first_model, _, source, destination = cobra_reaction(reaction_id="two")
    one = Reaction("one")
    one.add_metabolites({source: -1, destination: 1})
    first_model.add_reactions([one])
    resolved = resolve_authoritative_transitions(first_model, [assignment("one"), assignment("two")])
    assert tuple(resolved) == ("one", "two")


CONTROL_0_REACTIONS = tuple(f"v{index}" for index in range(1, 9))
CONTROL_0_TRANSITIONS = (
    "antoniewicz.table5.v1.citrate_synthase",
    "antoniewicz.table5.v2.citrate_to_akg",
    "antoniewicz.table5.v3.akg_to_glutamate",
    "antoniewicz.table5.v4.akg_to_succinate",
    "antoniewicz.table5.v5.succinate_to_fumarate",
    "antoniewicz.table5.v6.fumarate_to_oaa",
    "antoniewicz.table5.v7.oaa_to_fumarate",
    "antoniewicz.table5.v8.aspartate_to_oaa",
)
GLYCOLYSIS_REACTIONS = (
    "GLC_IN", "HEX", "PGI", "PFK", "FBA", "TPI", "GAPD", "PGK", "PGM",
    "ENO", "PYK", "LDH", "PDH",
)
GLYCOLYSIS_TRANSITIONS = (
    "transport.glucose.identity",
    "glycolysis.hexokinase",
    "glycolysis.phosphoglucose_isomerase",
    "glycolysis.phosphofructokinase",
    "glycolysis.fructose_bisphosphate_aldolase",
    "glycolysis.triose_phosphate_isomerase",
    "glycolysis.gap_to_bpg",
    "glycolysis.bpg_to_3pg",
    "glycolysis.3pg_to_2pg",
    "glycolysis.2pg_to_pep",
    "glycolysis.pyruvate_kinase",
    "pyruvate.lactate_dehydrogenase",
    "pyruvate.pyruvate_dehydrogenase",
)


def _assignments(reactions, transitions):
    return tuple(Assignment(reaction, transition, "forward") for reaction, transition in zip(reactions, transitions))


def _assert_authoritative_projection(cobra_model, canonical, assignments):
    resolved = resolve_authoritative_transitions(cobra_model, assignments, require_complete=True)
    isotope_by_id = {item.reaction_id: item for item in canonical.isotope_model.reactions}
    for assignment_item in assignments:
        actual = isotope_by_id[assignment_item.reaction_id]
        selected = resolved[assignment_item.reaction_id]
        transition = selected.transition
        source_binding = dict(zip(transition.substrates, selected.substrate_ids))
        destination_binding = dict(zip(transition.products, selected.product_ids))
        assert actual.directional_id == f"{transition.canonical_id}:forward"
        assert actual.direction == "forward" and actual.isotope_enabled
        assert tuple(x.metabolite_id for x in actual.substrates) == selected.substrate_ids
        assert tuple(x.metabolite_id for x in actual.products) == selected.product_ids
        for participant, canonical_id in zip(actual.substrates, transition.substrates):
            assert participant.atom_positions == tuple(range(1, LIBRARY.metabolites[canonical_id].carbon_count + 1))
        for participant, canonical_id in zip(actual.products, transition.products):
            assert participant.atom_positions == tuple(range(1, LIBRARY.metabolites[canonical_id].carbon_count + 1))
        expected_branches = transition.forward_branches
        assert tuple((x.branch_id, x.weight) for x in actual.mapping_branches) == tuple(
            (x.branch_id, x.weight) for x in expected_branches
        )
        for actual_branch, expected_branch in zip(actual.mapping_branches, expected_branches):
            assert tuple(
                (x.source.metabolite_id, x.source.position, x.destination.metabolite_id, x.destination.position)
                for x in actual_branch.transitions
            ) == tuple(
                (source_binding[x.source.metabolite], x.source.position,
                 destination_binding[x.destination.metabolite], x.destination.position)
                for x in expected_branch.atom_map
            )
        assert actual.symmetry_semantics == transition.symmetry
        assert actual.provenance == tuple(transition.provenance.to_dict().items()) + (
            ("validation_status", transition.validation_status),
        )


@pytest.mark.parametrize(
    ("directory", "reaction_ids", "transition_ids", "metabolite_ids", "unbalanced", "counts", "objective"),
    [
        (
            "antoniewicz_tca", CONTROL_0_REACTIONS, CONTROL_0_TRANSITIONS,
            ("OAC", "AcCoA", "citrate", "AKG", "glutamate", "succinate", "fumarate", "aspartate", "CO2"),
            {"AcCoA", "aspartate", "CO2", "glutamate"},
            (4, 2, 6, 5, 5, 4, 4, 4, 1), "v3",
        ),
        (
            "antoniewicz_tca_glucose", GLYCOLYSIS_REACTIONS + CONTROL_0_REACTIONS,
            GLYCOLYSIS_TRANSITIONS + CONTROL_0_TRANSITIONS,
            ("glucose_ext", "glucose_c", "G6P", "F6P", "FBP", "DHAP", "GAP", "BPG", "3PG", "2PG", "PEP", "pyruvate", "lactate", "AcCoA", "OAC", "citrate", "AKG", "glutamate", "succinate", "fumarate", "aspartate", "CO2"),
            {"glucose_ext", "aspartate", "CO2", "lactate", "glutamate"},
            (6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 6, 5, 5, 4, 4, 4, 1), "v1",
        ),
    ],
)
def test_real_controls_exact_canonical_projection(
    directory, reaction_ids, transition_ids, metabolite_ids, unbalanced, counts, objective
):
    cobra_model = read_sbml_model(f"codex/examples/{directory}/{directory}.xml")
    assignments = _assignments(reaction_ids, transition_ids)
    canonical = project_cobra_model(cobra_model, assignments, require_complete=True)
    assert tuple(x.reaction_id for x in canonical.flux_model.reactions) == reaction_ids
    assert all((x.lower_bound, x.upper_bound) == (0.0, 1_000_000.0) for x in canonical.flux_model.reactions)
    assert canonical.flux_model.objective.direction == "maximise"
    assert tuple((x.reaction_id, x.coefficient) for x in canonical.flux_model.objective.terms) == ((objective, 1.0),)
    assert tuple(x.metabolite_id for x in canonical.flux_model.metabolites) == metabolite_ids
    assert {x.metabolite_id for x in canonical.flux_model.metabolites if not x.steady_state_balanced} == unbalanced
    assert tuple(x.carbon_count for x in canonical.isotope_model.metabolites) == counts
    assert all(x.isotope_visible for x in canonical.isotope_model.metabolites)
    assert {x.metabolite_id for x in canonical.isotope_model.metabolites if x.symmetry} == {"succinate", "fumarate"}
    _assert_authoritative_projection(cobra_model, canonical, assignments)
    branches = {x.reaction_id: tuple((b.branch_id, b.weight) for b in x.mapping_branches) for x in canonical.isotope_model.reactions}
    assert branches["v5"] == (("canonical_orientation", 0.5), ("reversed_orientation", 0.5))
    expected_fumarate = (("canonical_orientation", 0.5), ("reversed_symmetric_fumarate", 0.5))
    assert branches["v6"] == expected_fumarate
    assert branches["v7"] == expected_fumarate
    validate_canonical_model(canonical)
    assert model_fingerprint(canonical) == model_fingerprint(canonical)


@pytest.mark.parametrize(
    ("directory", "reaction_ids", "transition_ids", "missing"),
    [
        *[("antoniewicz_tca", CONTROL_0_REACTIONS, CONTROL_0_TRANSITIONS, item) for item in ("v5", "v6", "v7")],
        *[("antoniewicz_tca_glucose", GLYCOLYSIS_REACTIONS + CONTROL_0_REACTIONS,
           GLYCOLYSIS_TRANSITIONS + CONTROL_0_TRANSITIONS, item) for item in ("v5", "v6", "v7", "FBA")],
    ],
)
def test_real_controls_complete_manifest_rejects_missing_authority(directory, reaction_ids, transition_ids, missing):
    cobra_model = read_sbml_model(f"codex/examples/{directory}/{directory}.xml")
    assignments = tuple(x for x in _assignments(reaction_ids, transition_ids) if x.reaction_id != missing)
    with pytest.raises(ProjectionError, match=rf"missing assignment.*{missing}"):
        project_cobra_model(cobra_model, assignments, require_complete=True)


@pytest.mark.parametrize(
    ("directory", "yaml_name", "reaction_ids", "transition_ids"),
    [
        ("antoniewicz_tca", "experiment_stationary.yaml", CONTROL_0_REACTIONS, CONTROL_0_TRANSITIONS),
        ("antoniewicz_tca_glucose", "experiment_u13c6_glucose.yaml", GLYCOLYSIS_REACTIONS + CONTROL_0_REACTIONS, GLYCOLYSIS_TRANSITIONS + CONTROL_0_TRANSITIONS),
    ],
)
def test_real_stationary_experiments_preserve_order_validate_and_fingerprint(
    directory, yaml_name, reaction_ids, transition_ids
):
    cobra_model = read_sbml_model(f"codex/examples/{directory}/{directory}.xml")
    canonical_model = project_cobra_model(cobra_model, _assignments(reaction_ids, transition_ids), require_complete=True)
    config = load_experiment(f"codex/examples/{directory}/{yaml_name}")
    experiment = project_stationary_experiment(config)
    assert tuple(x.metabolite_id for x in experiment.tracers) == tuple(x.metabolite_id for x in config.tracers)
    assert tuple(x.isotopomers for x in experiment.tracers) == tuple(tuple(x.isotopomer_fractions.items()) for x in config.tracers)
    assert tuple(x.target_id for x in experiment.targets) == tuple(x.fragment_id for x in config.targets)
    assert tuple(x.atom_positions for x in experiment.targets) == tuple(x.atom_positions for x in config.targets)
    validate_stationary_experiment(canonical_model, experiment)
    assert experiment_fingerprint(canonical_model, experiment) == experiment_fingerprint(canonical_model, project_stationary_experiment(config))
    assert experiment_fingerprint(canonical_model, experiment) != model_fingerprint(canonical_model)
