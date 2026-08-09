"""Focused acceptance tests for authoritative COBRA compatibility projection."""

from __future__ import annotations

import pytest
from cobra import Metabolite, Model, Reaction

from fluxemu.carbon_transitions import load_default_library
from fluxemu.compat import (
    AuthoritativeTransitionAssignment as Assignment,
    ProjectionError,
    project_cobra_model,
    resolve_authoritative_transitions,
)
from fluxemu.isotope_metadata import (
    AtomMappedParticipant,
    MetaboliteIsotopeMetadata,
    ReactionIsotopeMetadata,
    set_metabolite_metadata,
    set_reaction_metadata,
)
from fluxemu.model import validate_canonical_model


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
