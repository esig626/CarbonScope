"""Convert normalized ``metabolite.C<n>`` maps to FluxEMU/mfapy structures."""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_lowercase
from typing import Literal, Mapping, Sequence

from fluxemu.exceptions import CarbonTransitionValidationError, MappingError
from fluxemu.isotope_metadata import AtomMappedParticipant, ReactionIsotopeMetadata

from .schema import AtomRef, CarbonTransition, MappingBranch, MetaboliteDefinition


@dataclass(frozen=True)
class MfapyAtomMapExport:
    """One mfapy-ready primary map plus its validated branch description.

    mfapy represents the Antoniewicz-style equal orientations through the
    ``symmetry`` attribute of affected metabolites.  ``branches`` is retained
    here so callers can report the exact probabilistic treatment instead of
    silently collapsing it to the primary string.
    """

    atommap: str
    substrate_labels: tuple[tuple[str, tuple[str, ...]], ...]
    product_labels: tuple[tuple[str, tuple[str, ...]], ...]
    branches: tuple[MappingBranch, ...]
    symmetry_metabolites: tuple[str, ...]


def _directed_sides(
    entry: CarbonTransition, direction: Literal["forward", "reverse"]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (entry.substrates, entry.products) if direction == "forward" else (entry.products, entry.substrates)


def _labels_for_map(
    entry: CarbonTransition,
    direction: Literal["forward", "reverse"],
    registry: Mapping[str, MetaboliteDefinition],
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    substrates, products = _directed_sides(entry, direction)
    branches = entry.branch_for_direction(direction)
    primary = branches[0].atom_map
    source_order = [
        AtomRef(metabolite, position)
        for metabolite in substrates
        for position in range(1, registry[metabolite].carbon_count + 1)
    ]
    if len(source_order) > len(ascii_lowercase):
        raise MappingError(
            f"{entry.canonical_id} has {len(source_order)} tracked carbons; "
            "mfapy export currently supports at most 26 unique atom labels"
        )
    labels = {atom: ascii_lowercase[index] for index, atom in enumerate(source_order)}
    destination_origins = {transition.destination: transition.source for transition in primary}
    substrate_labels = tuple(
        (metabolite, tuple(labels[AtomRef(metabolite, position)] for position in range(1, registry[metabolite].carbon_count + 1)))
        for metabolite in substrates
    )
    product_labels = tuple(
        (
            metabolite,
            tuple(
                labels[destination_origins[AtomRef(metabolite, position)]]
                for position in range(1, registry[metabolite].carbon_count + 1)
            ),
        )
        for metabolite in products
    )
    return substrate_labels, product_labels


def export_mfapy_atom_map(
    entry: CarbonTransition,
    registry: Mapping[str, MetaboliteDefinition],
    *,
    direction: Literal["forward", "reverse"] = "forward",
) -> MfapyAtomMapExport:
    """Export a validated normalized entry without making it canonical data.

    Equal mapping branches are supported through the same symmetric-metabolite
    mechanism that reproduces the frozen Antoniewicz Table 5 benchmark.  A
    branch involving no declared symmetric metabolite is rejected, because
    mfapy has no lossless single-reaction syntax for arbitrary probabilities.
    """

    branches = entry.branch_for_direction(direction)
    if len(branches) > 1:
        symmetric = {
            metabolite
            for metabolite in (*entry.substrates, *entry.products)
            if registry[metabolite].symmetry
        }
        if not symmetric:
            raise MappingError(
                f"{entry.canonical_id} has weighted branches that cannot be represented by mfapy without a symmetric metabolite"
            )
    substrate_labels, product_labels = _labels_for_map(entry, direction, registry)
    atommap = "+".join("".join(labels) for _, labels in substrate_labels)
    atommap += "-->"
    atommap += "+".join("".join(labels) for _, labels in product_labels)
    symmetric = tuple(
        metabolite
        for metabolite in (*entry.substrates, *entry.products)
        if registry[metabolite].symmetry
    )
    return MfapyAtomMapExport(atommap, substrate_labels, product_labels, branches, symmetric)


def export_reaction_metadata(
    entry: CarbonTransition,
    registry: Mapping[str, MetaboliteDefinition],
    *,
    cobra_reaction_id: str,
    substrate_ids: Sequence[str],
    product_ids: Sequence[str],
    direction: Literal["forward", "reverse"] = "forward",
    directional_id: str | None = None,
) -> ReactionIsotopeMetadata:
    """Generate FluxEMU SBML metadata for one matched model reaction.

    ``substrate_ids`` and ``product_ids`` are positional to the entry's
    directed carbon sides.  Keeping them positional permits identity transport
    maps where a canonical metabolite occurs in both compartments.
    """

    directed_substrates, directed_products = _directed_sides(entry, direction)
    if len(substrate_ids) != len(directed_substrates) or len(product_ids) != len(directed_products):
        raise MappingError(
            f"{entry.canonical_id} model participants do not match its carbon reaction arity"
        )
    exported = export_mfapy_atom_map(entry, registry, direction=direction)
    substrate_participants = tuple(
        AtomMappedParticipant(model_id, labels)
        for model_id, (_, labels) in zip(substrate_ids, exported.substrate_labels)
    )
    product_participants = tuple(
        AtomMappedParticipant(model_id, labels)
        for model_id, (_, labels) in zip(product_ids, exported.product_labels)
    )
    return ReactionIsotopeMetadata(
        original_cobra_reaction_id=cobra_reaction_id,
        directional_id=directional_id or f"library:{entry.canonical_id}:{direction}",
        direction=direction,
        include_in_isotope_model=True,
        substrates=substrate_participants,
        products=product_participants,
    )


__all__ = ["MfapyAtomMapExport", "export_mfapy_atom_map", "export_reaction_metadata"]
