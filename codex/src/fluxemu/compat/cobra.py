"""Authoritative, one-way projection of COBRApy state into FluxEMU.

Stoichiometry is used only to verify an explicitly selected transition.  It is
never used to discover a transition or manufacture atom mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

from cobra.util.solver import linear_reaction_coefficients

from fluxemu.carbon_transitions import (
    ResolvedTransition,
    TransitionLibrary,
    load_default_library,
    validate_transition,
)
from fluxemu.isotope_metadata import (
    ReactionIsotopeMetadata,
    collect_isotope_metadata,
)
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
    StoichiometricTerm,
    validate_canonical_model,
    validate_flux_model,
)


class ProjectionError(ValueError):
    """External state cannot be projected without inventing science."""


@dataclass(frozen=True, slots=True)
class AuthoritativeTransitionAssignment:
    """Internal migration binding from one external reaction to library state."""

    reaction_id: str
    transition_id: str
    direction: Literal["forward", "reverse"]


def _reaction_index(model) -> dict[str, object]:
    return {reaction.id: reaction for reaction in model.reactions}


def _canonical_side(library: TransitionLibrary, reaction, sign: int):
    result = []
    for metabolite, coefficient in reaction.metabolites.items():
        if float(coefficient) * sign <= 0:
            continue
        canonical_id = library.metabolite_id(metabolite)
        if canonical_id is not None:
            result.append((canonical_id, metabolite.id))
    return tuple(result)


def _bind_side(
    reaction_id: str,
    side_name: str,
    expected: tuple[str, ...],
    observed: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    if len(expected) != len(observed):
        raise ProjectionError(
            f"reaction '{reaction_id}' {side_name} participant arity does not match the authoritative transition"
        )
    bound = []
    for canonical_id, (observed_canonical, model_id) in zip(expected, observed):
        if canonical_id != observed_canonical:
            raise ProjectionError(
                f"reaction '{reaction_id}' {side_name} participant mismatch: "
                f"expected '{canonical_id}', found '{model_id}' ({observed_canonical})"
            )
        bound.append(model_id)
    return tuple(bound)


def _verify_compartments(reaction, transition, substrate_ids, product_ids) -> None:
    metabolites = reaction.model.metabolites
    substrate_compartments = tuple(metabolites.get_by_id(item).compartment for item in substrate_ids)
    product_compartments = tuple(metabolites.get_by_id(item).compartment for item in product_ids)
    shared = set(transition.substrates) & set(transition.products)
    if shared:
        compatible = len(transition.substrates) == len(transition.products) == 1 and (
            substrate_compartments[0] != product_compartments[0]
        )
    else:
        compatible = len(set((*substrate_compartments, *product_compartments))) <= 1
    if not compatible:
        raise ProjectionError(f"reaction '{reaction.id}' compartments do not match the authoritative transition")


def resolve_authoritative_transitions(
    cobra_model,
    assignments: Iterable[AuthoritativeTransitionAssignment],
    *,
    library: TransitionLibrary | None = None,
    require_complete: bool = False,
) -> Mapping[str, ResolvedTransition]:
    """Resolve only direct, explicit transition IDs, retaining assignment order."""

    transition_library = library or load_default_library()
    reactions = _reaction_index(cobra_model)
    metadata = collect_isotope_metadata(cobra_model)
    resolved: dict[str, ResolvedTransition] = {}
    for assignment in tuple(assignments):
        if not isinstance(assignment, AuthoritativeTransitionAssignment):
            raise ProjectionError("authoritative assignments must be AuthoritativeTransitionAssignment records")
        if assignment.reaction_id in resolved:
            raise ProjectionError(f"duplicate authoritative assignment for reaction '{assignment.reaction_id}'")
        reaction = reactions.get(assignment.reaction_id)
        if reaction is None:
            raise ProjectionError(f"authoritative assignment references unknown reaction '{assignment.reaction_id}'")
        try:
            transition = transition_library.by_id[assignment.transition_id]
        except KeyError as error:
            raise ProjectionError(f"unknown authoritative transition ID '{assignment.transition_id}'") from error
        if assignment.direction not in {"forward", "reverse"}:
            raise ProjectionError(f"reaction '{assignment.reaction_id}' has invalid explicit direction {assignment.direction!r}")
        try:
            validate_transition(transition, transition_library.metabolites)
            transition.branch_for_direction(assignment.direction)
        except Exception as error:
            raise ProjectionError(
                f"reaction '{assignment.reaction_id}' cannot use transition '{assignment.transition_id}' "
                f"in {assignment.direction} direction: {error}"
            ) from error

        observed_substrates = _canonical_side(transition_library, reaction, -1)
        observed_products = _canonical_side(transition_library, reaction, 1)
        expected_substrates = transition.substrates if assignment.direction == "forward" else transition.products
        expected_products = transition.products if assignment.direction == "forward" else transition.substrates
        substrate_ids = _bind_side(reaction.id, "substrate", expected_substrates, observed_substrates)
        product_ids = _bind_side(reaction.id, "product", expected_products, observed_products)
        _verify_compartments(reaction, transition, substrate_ids, product_ids)
        resolved[reaction.id] = ResolvedTransition(
            reaction.id,
            transition,
            assignment.direction,
            "explicit_id",
            substrate_ids,
            product_ids,
        )

    if require_complete:
        expected = tuple(item.original_cobra_reaction_id for item in metadata.included_reactions)
        missing = tuple(item for item in expected if item not in resolved)
        if missing:
            raise ProjectionError("complete authoritative manifest is missing assignment(s): " + ", ".join(missing))
        extra = tuple(item for item in resolved if item not in expected)
        if extra:
            raise ProjectionError("complete authoritative manifest has extra assignment(s): " + ", ".join(extra))
    return MappingProxyType(resolved)


def _participants(model_ids, canonical_ids, library):
    return tuple(
        IsotopeParticipant(model_id, tuple(range(1, library.metabolites[canonical_id].carbon_count + 1)))
        for model_id, canonical_id in zip(model_ids, canonical_ids)
    )


def _authoritative_reaction(resolved: ResolvedTransition, library: TransitionLibrary) -> IsotopeReaction:
    transition = resolved.transition
    if resolved.direction == "forward":
        source_names, destination_names = transition.substrates, transition.products
    else:
        source_names, destination_names = transition.products, transition.substrates
    source_binding = dict(zip(source_names, resolved.substrate_ids))
    destination_binding = dict(zip(destination_names, resolved.product_ids))
    branches = tuple(
        MappingBranch(
            branch.branch_id,
            branch.weight,
            tuple(
                AtomTransition(
                    AtomPosition(source_binding[item.source.metabolite], item.source.position),
                    AtomPosition(destination_binding[item.destination.metabolite], item.destination.position),
                )
                for item in branch.atom_map
            ),
        )
        for branch in transition.branch_for_direction(resolved.direction)
    )
    provenance = transition.provenance.to_dict()
    return IsotopeReaction(
        resolved.model_reaction_id,
        resolved.direction,
        True,
        _participants(resolved.substrate_ids, source_names, library),
        _participants(resolved.product_ids, destination_names, library),
        branches,
        f"{transition.canonical_id}:{resolved.direction}",
        transition.symmetry,
        tuple(provenance.items()) + (("validation_status", transition.validation_status),),
    )


def _legacy_reaction(reaction, metadata: ReactionIsotopeMetadata, metabolite_metadata) -> IsotopeReaction:
    symmetric = any(
        metabolite_metadata[item.metabolite_id].symmetry
        for item in metadata.substrates + metadata.products
        if item.metabolite_id in metabolite_metadata
    )
    if symmetric:
        raise ProjectionError(
            f"reaction '{reaction.id}' has only a lossy primary map for Boolean symmetry; "
            "authoritative weighted transition information is required"
        )
    if not metadata.substrates or not metadata.products:
        raise ProjectionError(f"reaction '{reaction.id}' is missing a complete explicit legacy atom map")
    negative_ids = tuple(item.id for item, coefficient in reaction.metabolites.items() if coefficient < 0)
    positive_ids = tuple(item.id for item, coefficient in reaction.metabolites.items() if coefficient > 0)
    expected_substrates = negative_ids if metadata.direction == "forward" else positive_ids
    expected_products = positive_ids if metadata.direction == "forward" else negative_ids
    if tuple(item.metabolite_id for item in metadata.substrates) != expected_substrates or tuple(
        item.metabolite_id for item in metadata.products
    ) != expected_products:
        raise ProjectionError(f"reaction '{reaction.id}' legacy participant identities or direction disagree with stoichiometry")
    product_atoms = {
        label: AtomPosition(participant.metabolite_id, position)
        for participant in metadata.products
        for position, label in enumerate(participant.atom_labels, 1)
    }
    transitions = []
    for participant in metadata.substrates:
        for position, label in enumerate(participant.atom_labels, 1):
            destination = product_atoms.get(label)
            if destination is None:
                raise ProjectionError(f"reaction '{reaction.id}' legacy map has incomplete destination coverage")
            transitions.append(AtomTransition(AtomPosition(participant.metabolite_id, position), destination))
    if len(transitions) != len(product_atoms) or len({item.destination for item in transitions}) != len(product_atoms):
        raise ProjectionError(f"reaction '{reaction.id}' legacy map is not a complete one-to-one atom map")
    for participant in metadata.substrates + metadata.products:
        definition = metabolite_metadata.get(participant.metabolite_id)
        if definition is None or definition.carbon_count != len(participant.atom_labels):
            raise ProjectionError(f"reaction '{reaction.id}' legacy participant carbon count mismatch")
    convert = lambda items: tuple(IsotopeParticipant(item.metabolite_id, tuple(range(1, len(item.atom_labels) + 1))) for item in items)
    return IsotopeReaction(
        reaction.id, metadata.direction, True, convert(metadata.substrates), convert(metadata.products),
        (MappingBranch("primary", 1.0, tuple(transitions)),), metadata.directional_id,
        "explicit non-symmetric legacy map", (("source", "legacy FluxEMU metadata"),),
    )


def project_cobra_model(
    cobra_model,
    assignments: Iterable[AuthoritativeTransitionAssignment] = (),
    *,
    library: TransitionLibrary | None = None,
    require_complete: bool = False,
) -> CanonicalModel:
    """Project COBRA order and explicit isotope chemistry into canonical state."""

    transition_library = library or load_default_library()
    metadata = collect_isotope_metadata(cobra_model)
    resolved = resolve_authoritative_transitions(
        cobra_model, assignments, library=transition_library, require_complete=require_complete
    )
    # Isotope-forward models deliberately retain their reviewed tracer-source
    # and terminal-pool boundary roles.  This is distinct from projecting the
    # COBRA LP, where every metabolite remains a steady-state constraint row.
    flux_model = _project_cobra_flux_model(cobra_model, isotope_boundary_roles=True)

    authoritative_metabolites = {}
    for match in resolved.values():
        source_names = match.transition.substrates if match.direction == "forward" else match.transition.products
        product_names = match.transition.products if match.direction == "forward" else match.transition.substrates
        authoritative_metabolites.update(zip(match.substrate_ids, source_names))
        authoritative_metabolites.update(zip(match.product_ids, product_names))
    isotope_metabolites = tuple(
        IsotopeMetabolite(
            item.id,
            transition_library.metabolites[authoritative_metabolites[item.id]].carbon_count
            if item.id in authoritative_metabolites
            else (metadata.metabolites[item.id].carbon_count if item.id in metadata.metabolites else 0),
            item.id in authoritative_metabolites
            or (metadata.metabolites[item.id].include_in_isotope_model if item.id in metadata.metabolites else False),
            transition_library.metabolites[authoritative_metabolites[item.id]].symmetry
            if item.id in authoritative_metabolites
            else (metadata.metabolites[item.id].symmetry if item.id in metadata.metabolites else False),
        )
        for item in cobra_model.metabolites
    )
    isotope_reactions = []
    for reaction in cobra_model.reactions:
        if reaction.id in resolved:
            isotope_reactions.append(_authoritative_reaction(resolved[reaction.id], transition_library))
        elif reaction.id in metadata.reactions and metadata.reactions[reaction.id].include_in_isotope_model:
            isotope_reactions.append(_legacy_reaction(reaction, metadata.reactions[reaction.id], metadata.metabolites))
        else:
            visible_sources = any(
                coefficient < 0
                and item.id in metadata.metabolites
                and metadata.metabolites[item.id].include_in_isotope_model
                for item, coefficient in reaction.metabolites.items()
            )
            visible_products = any(
                coefficient > 0
                and item.id in metadata.metabolites
                and metadata.metabolites[item.id].include_in_isotope_model
                for item, coefficient in reaction.metabolites.items()
            )
            if visible_sources and visible_products:
                raise ProjectionError(
                    f"reaction '{reaction.id}' connects isotope-visible participants but is missing an explicit legacy map or authoritative assignment"
                )
            isotope_reactions.append(IsotopeReaction(reaction.id, "forward", False, (), (), ()))
    model = CanonicalModel(
        flux_model,
        IsotopeModel(isotope_metabolites, tuple(isotope_reactions)),
    )
    validate_canonical_model(model)
    return model


def project_cobra_flux_model(cobra_model) -> FluxModel:
    """Faithfully project the COBRA LP, independent of isotope annotations."""

    return _project_cobra_flux_model(cobra_model, isotope_boundary_roles=False)


def _project_cobra_flux_model(cobra_model, *, isotope_boundary_roles: bool) -> FluxModel:
    """Project constraints under an explicit, non-conflated balance policy."""

    metadata = collect_isotope_metadata(cobra_model) if isotope_boundary_roles else None
    flux_metabolites = tuple(
        FluxMetabolite(
            item.id,
            not (
                isotope_boundary_roles
                and item.id in metadata.metabolites
                and (metadata.metabolites[item.id].is_carbon_source or metadata.metabolites[item.id].is_excreted)
            ),
        )
        for item in cobra_model.metabolites
    )
    flux_reactions = tuple(
        FluxReaction(
            reaction.id,
            tuple(StoichiometricTerm(item.id, coefficient) for item, coefficient in reaction.metabolites.items()),
            reaction.lower_bound,
            reaction.upper_bound,
        )
        for reaction in cobra_model.reactions
    )
    coefficients = linear_reaction_coefficients(cobra_model)
    objective = LinearObjective(
        "maximise" if cobra_model.objective.direction == "max" else "minimise",
        tuple(ObjectiveTerm(reaction.id, coefficients[reaction]) for reaction in cobra_model.reactions if reaction in coefficients),
    )
    model = FluxModel(flux_metabolites, flux_reactions, objective)
    validate_flux_model(model)
    return model
