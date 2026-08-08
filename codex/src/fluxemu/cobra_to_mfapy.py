"""Validated in-memory construction of mfapy objects from an annotated model."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

from cobra import Model

from ._mfapy import load_mfapy
from .configuration import ExperimentConfig
from .exceptions import ConfigurationError, ForwardEMUError, MappingError, MetadataError
from .flux_conversion import FluxConverter, ReactionMapping
from .isotope_metadata import (
    AtomMappedParticipant,
    IsotopeMetadataCollection,
    ReactionIsotopeMetadata,
)


_SAFE_PREFIXES = {
    "reaction": "fr",
    "metabolite": "fm",
    "target": "ft",
    "reversible": "fv",
}


def deterministic_internal_id(kind: str, original_id: str) -> str:
    """Return a conservative deterministic identifier safe in generated code."""

    if kind not in _SAFE_PREFIXES:
        raise MappingError(f"unknown internal identifier kind '{kind}'")
    if not isinstance(original_id, str) or not original_id:
        raise MappingError("cannot generate an internal ID for an empty identifier")
    digest = sha256(original_id.encode("utf-8")).hexdigest()[:16]
    return f"{_SAFE_PREFIXES[kind]}{digest}"


@dataclass(frozen=True)
class MfapyModelBundle:
    """One constructed mfapy model plus all one-to-one ID mappings."""

    model: Any
    converter: FluxConverter
    reaction_mappings: tuple[ReactionMapping, ...]
    metabolite_to_internal: Mapping[str, str]
    target_to_internal: Mapping[str, str]
    reactions: Mapping[str, Mapping[str, Any]]
    reversible_reactions: Mapping[str, Mapping[str, Any]]
    metabolites: Mapping[str, Mapping[str, Any]]
    target_fragments: Mapping[str, Mapping[str, Any]]

    @property
    def internal_to_target(self) -> dict[str, str]:
        return {internal: original for original, internal in self.target_to_internal.items()}


def _register_ids(kind: str, original_ids: list[str]) -> dict[str, str]:
    generated: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for original_id in original_ids:
        if original_id in generated:
            raise MappingError(f"duplicate {kind} identifier '{original_id}'")
        internal_id = deterministic_internal_id(kind, original_id)
        if internal_id in reverse:
            raise MappingError(
                f"generated {kind} ID collision between '{reverse[internal_id]}' "
                f"and '{original_id}'"
            )
        generated[original_id] = internal_id
        reverse[internal_id] = original_id
    return generated


def _require_complete_metadata(
    cobra_model: Model, metadata: IsotopeMetadataCollection
) -> None:
    reaction_ids = {reaction.id for reaction in cobra_model.reactions}
    metabolite_ids = {metabolite.id for metabolite in cobra_model.metabolites}
    missing_reactions = sorted(reaction_ids - set(metadata.reactions))
    missing_metabolites = sorted(metabolite_ids - set(metadata.metabolites))
    if missing_reactions or missing_metabolites:
        parts = []
        if missing_reactions:
            parts.append("reactions=" + ",".join(missing_reactions))
        if missing_metabolites:
            parts.append("metabolites=" + ",".join(missing_metabolites))
        raise MetadataError(
            "every SBML network object must explicitly declare FluxEMU inclusion; "
            + "; ".join(parts)
        )


def _validate_participants(
    cobra_model: Model,
    reaction_metadata: ReactionIsotopeMetadata,
    metadata: IsotopeMetadataCollection,
) -> None:
    reaction = cobra_model.reactions.get_by_id(
        reaction_metadata.original_cobra_reaction_id
    )
    if reaction_metadata.direction == "forward":
        negative = {
            met.id
            for met, coefficient in reaction.metabolites.items()
            if coefficient < 0
        }
        positive = {
            met.id
            for met, coefficient in reaction.metabolites.items()
            if coefficient > 0
        }
    else:
        negative = {
            met.id
            for met, coefficient in reaction.metabolites.items()
            if coefficient > 0
        }
        positive = {
            met.id
            for met, coefficient in reaction.metabolites.items()
            if coefficient < 0
        }
    substrate_ids = {item.metabolite_id for item in reaction_metadata.substrates}
    product_ids = {item.metabolite_id for item in reaction_metadata.products}
    if not substrate_ids.issubset(negative):
        raise MappingError(
            f"reaction '{reaction.id}' substrate participants are not a subset of "
            "the COBRA-side substrates implied by direction"
        )
    if not product_ids.issubset(positive):
        raise MappingError(
            f"reaction '{reaction.id}' product participants are not a subset of "
            "the COBRA-side products implied by direction"
        )
    for coefficient in reaction.metabolites.values():
        if abs(float(coefficient)) != 1.0:
            raise MappingError(
                f"reaction '{reaction.id}' uses a non-unit stoichiometric coefficient; "
                "the first prototype requires explicit molecule-level directions"
            )

    for participant in (*reaction_metadata.substrates, *reaction_metadata.products):
        if participant.metabolite_id not in metadata.metabolites:
            raise MetadataError(
                f"reaction '{reaction.id}' references unannotated metabolite "
                f"'{participant.metabolite_id}'"
            )
        metabolite_metadata = metadata.metabolites[participant.metabolite_id]
        if not metabolite_metadata.include_in_isotope_model:
            raise MetadataError(
                f"reaction '{reaction.id}' tracks excluded metabolite "
                f"'{participant.metabolite_id}'"
            )
        if len(participant.atom_labels) != metabolite_metadata.carbon_count:
            raise MetadataError(
                f"reaction '{reaction.id}' atom count for '{participant.metabolite_id}' "
                "does not match metabolite metadata"
            )
        if any(len(label) != 1 for label in participant.atom_labels):
            raise MetadataError(
                f"reaction '{reaction.id}' uses a multi-character atom label; "
                "mfapy 0.6.3 requires single-character labels"
            )
    substrate_labels = set(reaction_metadata.substrate_atom_labels)
    missing_product_labels = set(reaction_metadata.product_atom_labels) - substrate_labels
    if missing_product_labels:
        raise MetadataError(
            f"reaction '{reaction.id}' product atom label(s) have no substrate origin: "
            + ", ".join(sorted(missing_product_labels))
        )


def _side_text(
    participants: tuple[AtomMappedParticipant, ...],
    metabolite_ids: Mapping[str, str],
) -> str:
    return "+".join(metabolite_ids[item.metabolite_id] for item in participants)


def _atom_side_text(participants: tuple[AtomMappedParticipant, ...]) -> str:
    return "+".join("".join(item.atom_labels) for item in participants)


def build_mfapy_model(
    cobra_model: Model,
    metadata: IsotopeMetadataCollection,
    experiment: ExperimentConfig,
) -> MfapyModelBundle:
    """Construct all four mfapy dictionaries and one reusable forward model."""

    if not isinstance(cobra_model, Model):
        raise MappingError("build_mfapy_model requires a cobra.Model")
    _require_complete_metadata(cobra_model, metadata)

    included_reaction_ids = [
        reaction.id
        for reaction in cobra_model.reactions
        if metadata.reactions[reaction.id].include_in_isotope_model
    ]
    included_metabolite_ids = [
        metabolite.id
        for metabolite in cobra_model.metabolites
        if metadata.metabolites[metabolite.id].include_in_isotope_model
    ]
    if not included_reaction_ids or not included_metabolite_ids:
        raise MetadataError("the isotope model must include reactions and metabolites")

    directional_ids = [
        metadata.reactions[rid].directional_id for rid in included_reaction_ids
    ]
    if len(directional_ids) != len(set(directional_ids)):
        raise MappingError("ambiguous directional identities in reaction metadata")

    reaction_ids = _register_ids("reaction", included_reaction_ids)
    metabolite_ids = _register_ids("metabolite", included_metabolite_ids)
    target_ids = _register_ids(
        "target", [target.fragment_id for target in experiment.targets]
    )

    metabolite_dictionary: dict[str, dict[str, Any]] = {}
    for order, original_id in enumerate(included_metabolite_ids):
        item = metadata.metabolites[original_id]
        metabolite_dictionary[metabolite_ids[original_id]] = {
            "C_number": item.carbon_count,
            "symmetry": "symmetry" if item.symmetry else "no",
            "carbonsource": "carbonsource" if item.is_carbon_source else "no",
            "excreted": "excreted" if item.is_excreted else "no",
            "order": order,
            "externalids": f"cobra:{original_id}",
            "lb": 0.0,
            "ub": 1_000_000.0,
        }

    reaction_dictionary: dict[str, dict[str, Any]] = {}
    mappings: list[ReactionMapping] = []
    for order, original_id in enumerate(included_reaction_ids):
        item = metadata.reactions[original_id]
        _validate_participants(cobra_model, item, metadata)
        reaction = cobra_model.reactions.get_by_id(original_id)
        substrate_text = _side_text(item.substrates, metabolite_ids)
        product_text = _side_text(item.products, metabolite_ids)
        if not substrate_text or not product_text:
            raise MetadataError(
                f"included reaction '{original_id}' must have tracked metabolites on both sides"
            )
        internal_id = reaction_ids[original_id]
        reaction_dictionary[internal_id] = {
            "stoichiometry": f"{substrate_text}-->{product_text}",
            "reaction": f"{substrate_text}-->{product_text}",
            "atommap": (
                f"{_atom_side_text(item.substrates)}-->"
                f"{_atom_side_text(item.products)}"
            ),
            "externalids": f"cobra:{original_id}",
            "order": order,
            "lb": max(0.0, float(reaction.lower_bound)),
            "ub": float(reaction.upper_bound),
        }
        mappings.append(
            ReactionMapping(
                original_cobra_reaction_id=original_id,
                internal_mfapy_reaction_id=internal_id,
                mfapy_reaction_order=order,
                direction=-1 if item.direction == "reverse" else 1,
            )
        )

    carbon_sources = {
        item.original_cobra_metabolite_id
        for item in metadata.included_metabolites
        if item.is_carbon_source
    }
    configured_sources = {tracer.metabolite_id for tracer in experiment.tracers}
    if configured_sources != carbon_sources:
        missing = sorted(carbon_sources - configured_sources)
        unexpected = sorted(configured_sources - carbon_sources)
        raise ConfigurationError(
            "tracers must exactly cover isotope carbon sources; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for tracer in experiment.tracers:
        if tracer.carbon_count != metadata.metabolites[tracer.metabolite_id].carbon_count:
            raise ConfigurationError(
                f"tracer '{tracer.metabolite_id}' isotopomer length does not match "
                "metabolite carbon count"
            )

    target_dictionary: dict[str, dict[str, Any]] = {}
    for order, target in enumerate(experiment.targets):
        if target.metabolite_id not in metadata.metabolites:
            raise ConfigurationError(
                f"target '{target.fragment_id}' references unknown metabolite "
                f"'{target.metabolite_id}'"
            )
        metabolite = metadata.metabolites[target.metabolite_id]
        if not metabolite.include_in_isotope_model:
            raise ConfigurationError(
                f"target '{target.fragment_id}' references an excluded metabolite"
            )
        if max(target.atom_positions) > metabolite.carbon_count:
            raise ConfigurationError(
                f"target '{target.fragment_id}' atom position exceeds metabolite carbon count"
            )
        if target.analytical_method not in {"intermediate", "gcms"}:
            raise ConfigurationError(
                f"target '{target.fragment_id}' analytical_method must be "
                "'intermediate' or 'gcms' for this prototype"
            )
        if target.correction != "no":
            raise ConfigurationError(
                "target-level natural-isotope correction is not supported by the "
                "forward-only prototype; use correction: no"
            )
        internal_target = target_ids[target.fragment_id]
        positions = ":".join(str(position) for position in target.atom_positions)
        target_dictionary[internal_target] = {
            "type": target.analytical_method,
            "atommap": f"{metabolite_ids[target.metabolite_id]}_{positions}",
            "use": "use",
            "order": order,
            "formula": target.formula,
        }

    # Keep fluxes in the exact order requested by the reaction dictionary.
    reversible_dictionary: dict[str, dict[str, Any]] = {}
    mfapy = load_mfapy()
    try:
        mfapy_model = mfapy.metabolicmodel.MetabolicModel(
            reaction_dictionary,
            reversible_dictionary,
            metabolite_dictionary,
            target_dictionary,
        )
    except Exception as error:
        raise ForwardEMUError(f"mfapy model construction failed: {error}") from error
    if not hasattr(mfapy_model, "func") or not callable(
        mfapy_model.func.get("calmdv") if isinstance(mfapy_model.func, dict) else None
    ):
        raise ForwardEMUError("mfapy returned an incomplete model without calmdv")

    internal_order = tuple(mfapy_model.reaction_ids)
    expected_order = tuple(reaction_ids[rid] for rid in included_reaction_ids)
    if internal_order != expected_order:
        raise MappingError(
            "mfapy reaction order differs from the constructed reaction mapping"
        )
    converter = FluxConverter(mappings, internal_order)
    return MfapyModelBundle(
        model=mfapy_model,
        converter=converter,
        reaction_mappings=tuple(mappings),
        metabolite_to_internal=dict(metabolite_ids),
        target_to_internal=dict(target_ids),
        reactions=reaction_dictionary,
        reversible_reactions=reversible_dictionary,
        metabolites=metabolite_dictionary,
        target_fragments=target_dictionary,
    )


__all__ = [
    "MfapyModelBundle",
    "build_mfapy_model",
    "deterministic_internal_id",
]
