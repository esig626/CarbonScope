"""Structural validation for normalized carbon-transition records."""

from __future__ import annotations

from math import isclose
from typing import Iterable, Mapping

from fluxemu.exceptions import CarbonTransitionValidationError

from .schema import AtomRef, AtomTransition, CarbonTransition, MappingBranch, MetaboliteDefinition


def _expected_atoms(metabolites: Iterable[str], registry: Mapping[str, MetaboliteDefinition]) -> set[AtomRef]:
    atoms: set[AtomRef] = set()
    for metabolite in metabolites:
        try:
            definition = registry[metabolite]
        except KeyError as error:
            raise CarbonTransitionValidationError(
                f"transition references metabolite '{metabolite}' absent from the metabolite registry"
            ) from error
        atoms.update(AtomRef(metabolite, position) for position in range(1, definition.carbon_count + 1))
    return atoms


def _validate_map(
    entry: CarbonTransition,
    atom_map: tuple[AtomTransition, ...],
    substrates: tuple[str, ...],
    products: tuple[str, ...],
    registry: Mapping[str, MetaboliteDefinition],
    context: str,
) -> None:
    expected_sources = _expected_atoms(substrates, registry)
    expected_destinations = _expected_atoms(products, registry)
    sources = [transition.source for transition in atom_map]
    destinations = [transition.destination for transition in atom_map]
    if len(sources) != len(set(sources)):
        raise CarbonTransitionValidationError(f"{entry.canonical_id} {context} gives a substrate carbon multiple fates")
    if len(destinations) != len(set(destinations)):
        raise CarbonTransitionValidationError(f"{entry.canonical_id} {context} gives a product carbon multiple origins")
    if set(sources) != expected_sources:
        missing = sorted(str(item) for item in expected_sources - set(sources))
        extra = sorted(str(item) for item in set(sources) - expected_sources)
        raise CarbonTransitionValidationError(
            f"{entry.canonical_id} {context} does not account for every substrate carbon; "
            f"missing={missing}, extra={extra}"
        )
    if set(destinations) != expected_destinations:
        missing = sorted(str(item) for item in expected_destinations - set(destinations))
        extra = sorted(str(item) for item in set(destinations) - expected_destinations)
        raise CarbonTransitionValidationError(
            f"{entry.canonical_id} {context} does not account for every product carbon; "
            f"missing={missing}, extra={extra}"
        )


def _map_signature(atom_map: tuple[AtomTransition, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(item.source), str(item.destination)) for item in atom_map))


def validate_transition(
    entry: CarbonTransition, registry: Mapping[str, MetaboliteDefinition]
) -> None:
    """Validate one entry's carbon bookkeeping, symmetry, and reverse map."""

    if not entry.substrates or not entry.products:
        raise CarbonTransitionValidationError(f"{entry.canonical_id} must have carbon substrates and products")
    if len(set(entry.substrates)) != len(entry.substrates) or len(set(entry.products)) != len(entry.products):
        raise CarbonTransitionValidationError(
            f"{entry.canonical_id} currently requires explicit unit stoichiometry for carbon metabolites"
        )
    participants = set(entry.substrates) | set(entry.products)
    if set(entry.carbon_counts) != participants:
        raise CarbonTransitionValidationError(
            f"{entry.canonical_id}.carbon_counts must cover exactly its carbon participants"
        )
    for metabolite, count in entry.carbon_counts.items():
        try:
            registered = registry[metabolite].carbon_count
        except KeyError as error:
            raise CarbonTransitionValidationError(
                f"{entry.canonical_id} uses unknown metabolite '{metabolite}'"
            ) from error
        if count != registered:
            raise CarbonTransitionValidationError(
                f"{entry.canonical_id} declares {count} carbons for {metabolite}; registry has {registered}"
            )

    _validate_map(entry, entry.forward_atom_map, entry.substrates, entry.products, registry, "forward_atom_map")
    branches = entry.forward_branches
    weights = sum(branch.weight for branch in branches)
    if not isclose(weights, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise CarbonTransitionValidationError(
            f"{entry.canonical_id} mapping branch weights must sum to 1, found {weights:g}"
        )
    branch_ids = [branch.branch_id for branch in branches]
    if len(branch_ids) != len(set(branch_ids)):
        raise CarbonTransitionValidationError(f"{entry.canonical_id} has duplicate mapping branch IDs")
    for branch in branches:
        _validate_map(entry, branch.atom_map, entry.substrates, entry.products, registry, f"branch '{branch.branch_id}'")

    if entry.reversible:
        if entry.reverse_atom_map == "not_applicable":
            raise CarbonTransitionValidationError(
                f"{entry.canonical_id} is reversible but declares reverse_atom_map: not_applicable"
            )
        reverse = entry.reverse_branches
        for branch in reverse:
            _validate_map(entry, branch.atom_map, entry.products, entry.substrates, registry, f"reverse branch '{branch.branch_id}'")
        recovered = tuple(branch.inverted() for branch in reverse)
        if len(recovered) != len(branches) or {
            (branch.branch_id, branch.weight, _map_signature(branch.atom_map)) for branch in recovered
        } != {
            (branch.branch_id, branch.weight, _map_signature(branch.atom_map)) for branch in branches
        }:
            raise CarbonTransitionValidationError(
                f"{entry.canonical_id} reverse mapping does not invert back to the forward map"
            )
    elif entry.reverse_atom_map != "not_applicable":
        raise CarbonTransitionValidationError(
            f"{entry.canonical_id} is irreversible and must declare reverse_atom_map: not_applicable"
        )

    if entry.mapping_branches and entry.symmetry == "none":
        raise CarbonTransitionValidationError(
            f"{entry.canonical_id} has weighted branches but does not describe its symmetry treatment"
        )


def validate_library(
    entries: Iterable[CarbonTransition], registry: Mapping[str, MetaboliteDefinition]
) -> None:
    """Validate the complete registry and reject duplicate canonical identities."""

    items = tuple(entries)
    canonical_ids = [entry.canonical_id for entry in items]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise CarbonTransitionValidationError("transition library has duplicate canonical_id values")
    aliases: dict[str, str] = {}
    for entry in items:
        validate_transition(entry, registry)
        for alias in entry.aliases:
            existing = aliases.setdefault(alias, entry.canonical_id)
            if existing != entry.canonical_id:
                raise CarbonTransitionValidationError(
                    f"model reaction alias '{alias}' is claimed by both {existing} and {entry.canonical_id}"
                )


__all__ = ["validate_library", "validate_transition"]
