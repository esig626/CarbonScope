"""Load the curated transition library and match it to COBRA reactions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence, TYPE_CHECKING

import yaml

from fluxemu.exceptions import MappingError, MetadataError
from fluxemu.isotope_metadata import (
    IsotopeMetadataCollection,
    MetaboliteIsotopeMetadata,
    ReactionIsotopeMetadata,
)

from .normalize import export_reaction_metadata
from .schema import CarbonTransition, MetaboliteDefinition
from .validator import validate_library


if TYPE_CHECKING:
    from cobra import Metabolite, Model, Reaction


DATA_FILENAMES = (
    "glycolysis.yaml",
    "pyruvate_lactate.yaml",
    "pentose_phosphate.yaml",
    "tca.yaml",
    "anaplerosis.yaml",
    "glutaminolysis.yaml",
    "malate_aspartate.yaml",
    "citrate_acetylcoa.yaml",
    "acetate.yaml",
    "serine_glycine.yaml",
)
EXPLICIT_ID_KEYS = ("fluxemu_transition_id", "fluxemu.transition_id")


def _yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise MappingError(f"could not parse transition data {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise MappingError(f"transition data {path} must be a YAML mapping")
    return value


def _values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _normal_forms(value: str) -> set[str]:
    result = {value.casefold()}
    stripped = re.sub(r"\[[A-Za-z][A-Za-z0-9_]*\]$", "", value)
    stripped = re.sub(r"_[A-Za-z][A-Za-z0-9]*$", "", stripped)
    result.add(stripped.casefold())
    return result


@dataclass(frozen=True)
class ResolvedTransition:
    """A library entry resolved to one directed COBRA reaction."""

    model_reaction_id: str
    transition: CarbonTransition
    direction: Literal["forward", "reverse"]
    matched_by: Literal["explicit_id", "database_identifier", "model_alias", "carbon_chemistry"]
    substrate_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MappingProvenanceRecord:
    """Auditable source decision for one carbon-carrying model reaction."""

    model_reaction_id: str
    canonical_transition_id: str | None
    mapping_source: Literal["SBML", "library", "unresolved"]
    source_identifier: str | None
    validation_status: str | None
    symmetry_treatment: str
    warning: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_reaction_id": self.model_reaction_id,
            "canonical_transition_id": self.canonical_transition_id,
            "mapping_source": self.mapping_source,
            "source_identifier": self.source_identifier,
            "validation_status": self.validation_status,
            "symmetry_treatment": self.symmetry_treatment,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class ResolvedMetadata:
    """Metadata plus a complete mapping-provenance report."""

    metadata: IsotopeMetadataCollection
    provenance: tuple[MappingProvenanceRecord, ...]


class TransitionLibrary:
    """Validated reaction and metabolite-carbon registry."""

    def __init__(
        self,
        metabolites: Iterable[MetaboliteDefinition],
        transitions: Iterable[CarbonTransition],
    ) -> None:
        metabolite_items = tuple(metabolites)
        metabolite_ids = [item.canonical_id for item in metabolite_items]
        if len(metabolite_ids) != len(set(metabolite_ids)):
            raise MappingError("metabolite carbon registry has duplicate canonical_id values")
        self.metabolites = MappingProxyType({item.canonical_id: item for item in metabolite_items})
        self.transitions = tuple(transitions)
        validate_library(self.transitions, self.metabolites)
        self.by_id = MappingProxyType({item.canonical_id: item for item in self.transitions})
        self._aliases = self._build_alias_index()
        self._identifiers = self._build_identifier_index()
        self._metabolite_aliases = self._build_metabolite_alias_index()

    @classmethod
    def from_directory(cls, directory: str | Path) -> "TransitionLibrary":
        root = Path(directory)
        metabolites_document = _yaml_mapping(root / "metabolites.yaml")
        if metabolites_document.get("schema_version") != 1:
            raise MappingError("metabolites.yaml must declare schema_version: 1")
        raw_metabolites = metabolites_document.get("metabolites")
        if not isinstance(raw_metabolites, list):
            raise MappingError("metabolites.yaml must contain a metabolites list")
        metabolites = tuple(
            MetaboliteDefinition.from_dict(item, f"metabolites[{index}]")
            for index, item in enumerate(raw_metabolites)
        )
        transitions: list[CarbonTransition] = []
        for filename in DATA_FILENAMES:
            path = root / filename
            if not path.is_file():
                raise MappingError(f"transition data file is missing: {path}")
            document = _yaml_mapping(path)
            if document.get("schema_version") != 1:
                raise MappingError(f"{filename} must declare schema_version: 1")
            raw_entries = document.get("entries")
            if not isinstance(raw_entries, list):
                raise MappingError(f"{filename} must contain an entries list")
            transitions.extend(
                CarbonTransition.from_dict(item, f"{filename}.entries[{index}]")
                for index, item in enumerate(raw_entries)
            )
        return cls(metabolites, transitions)

    def _build_alias_index(self) -> Mapping[str, CarbonTransition]:
        result: dict[str, CarbonTransition] = {}
        for entry in self.transitions:
            for alias in entry.aliases:
                for normal in _normal_forms(alias):
                    existing = result.setdefault(normal, entry)
                    if existing is not entry:
                        raise MappingError(f"reaction alias '{alias}' is ambiguous")
        return MappingProxyType(result)

    def _build_identifier_index(self) -> Mapping[str, tuple[CarbonTransition, ...]]:
        result: dict[str, list[CarbonTransition]] = defaultdict(list)
        for entry in self.transitions:
            for values in entry.identifiers.values():
                for value in values:
                    result[value.casefold()].append(entry)
        return MappingProxyType({key: tuple(value) for key, value in result.items()})

    def _build_metabolite_alias_index(self) -> Mapping[str, tuple[str, ...]]:
        result: dict[str, set[str]] = defaultdict(set)
        for definition in self.metabolites.values():
            values = (definition.canonical_id, definition.name, *definition.synonyms)
            for identifier_values in definition.identifiers.values():
                values += identifier_values
            for value in values:
                for normal in _normal_forms(value):
                    result[normal].add(definition.canonical_id)
        return MappingProxyType({key: tuple(sorted(value)) for key, value in result.items()})

    def metabolite_id(self, metabolite: "Metabolite") -> str | None:
        candidates: set[str] = set()
        for value in (metabolite.id, *[item for annotation in metabolite.annotation.values() for item in _values(annotation)]):
            for normal in _normal_forms(value):
                candidates.update(self._metabolite_aliases.get(normal, ()))
        if len(candidates) > 1:
            raise MappingError(
                f"model metabolite '{metabolite.id}' has ambiguous canonical identities: {sorted(candidates)}"
            )
        return next(iter(candidates), None)

    def _carbon_sides(self, reaction: "Reaction") -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
        substrates: list[tuple[str, str]] = []
        products: list[tuple[str, str]] = []
        unknown_carbon: list[str] = []
        for metabolite, coefficient in reaction.metabolites.items():
            canonical = self.metabolite_id(metabolite)
            if canonical is None:
                formula = metabolite.formula or ""
                if re.search(r"(?:^|[^A-Za-z])C(?:\d|$)", formula):
                    unknown_carbon.append(metabolite.id)
                continue
            if float(coefficient) == 0.0:
                continue
            target = substrates if coefficient < 0 else products
            copies = int(abs(float(coefficient)))
            if copies != abs(float(coefficient)) or copies != 1:
                raise MappingError(
                    f"reaction '{reaction.id}' has unsupported carbon-metabolite coefficient for '{metabolite.id}'"
                )
            target.append((canonical, metabolite.id))
        return tuple(substrates), tuple(products), tuple(unknown_carbon)

    @staticmethod
    def _ordered_ids(required: tuple[str, ...], observed: tuple[tuple[str, str], ...]) -> tuple[str, ...] | None:
        slots: dict[str, list[str]] = defaultdict(list)
        for canonical, model_id in observed:
            slots[canonical].append(model_id)
        if Counter(required) != Counter(canonical for canonical, _ in observed):
            return None
        return tuple(slots[canonical].pop(0) for canonical in required)

    def _orient(
        self,
        entry: CarbonTransition,
        substrates: tuple[tuple[str, str], ...],
        products: tuple[tuple[str, str], ...],
    ) -> tuple[Literal["forward", "reverse"], tuple[str, ...], tuple[str, ...]] | None:
        forward_substrates = self._ordered_ids(entry.substrates, substrates)
        forward_products = self._ordered_ids(entry.products, products)
        if forward_substrates is not None and forward_products is not None:
            return "forward", forward_substrates, forward_products
        if entry.reversible:
            reverse_substrates = self._ordered_ids(entry.products, substrates)
            reverse_products = self._ordered_ids(entry.substrates, products)
            if reverse_substrates is not None and reverse_products is not None:
                return "reverse", reverse_substrates, reverse_products
        return None

    @staticmethod
    def _compartments_compatible(
        reaction: "Reaction",
        entry: CarbonTransition,
        substrate_ids: tuple[str, ...],
        product_ids: tuple[str, ...],
    ) -> bool:
        """Require one biochemical compartment or an explicit identity transport.

        Ignoring compartment labels would let a reaction that accidentally joins
        cytosolic and mitochondrial carbon pools match an ordinary enzyme.  A
        canonical metabolite present on both sides is therefore treated as an
        identity transport record and must cross compartments; all other V1
        transitions require their recognized carbon metabolites to co-locate.
        """

        substrate_compartments = tuple(
            reaction.model.metabolites.get_by_id(identifier).compartment
            for identifier in substrate_ids
        )
        product_compartments = tuple(
            reaction.model.metabolites.get_by_id(identifier).compartment
            for identifier in product_ids
        )
        shared = set(entry.substrates) & set(entry.products)
        if shared:
            return len(entry.substrates) == len(entry.products) == 1 and (
                substrate_compartments[0] != product_compartments[0]
            )
        return len(set((*substrate_compartments, *product_compartments))) == 1

    def _candidate_from_annotation(self, reaction: "Reaction") -> CarbonTransition | None:
        explicit = [value for key in EXPLICIT_ID_KEYS for value in _values(reaction.annotation.get(key))]
        if not explicit:
            return None
        if len(explicit) != 1:
            raise MappingError(f"reaction '{reaction.id}' has ambiguous FluxEMU transition IDs")
        try:
            return self.by_id[explicit[0]]
        except KeyError as error:
            raise MappingError(
                f"reaction '{reaction.id}' explicitly requests unknown transition '{explicit[0]}'"
            ) from error

    def resolve_reaction(self, reaction: "Reaction") -> ResolvedTransition | None:
        """Resolve one reaction by the documented strict priority order."""

        substrates, products, unknown_carbon = self._carbon_sides(reaction)
        explicit = self._candidate_from_annotation(reaction)
        candidate_groups: tuple[tuple[str, tuple[CarbonTransition, ...]], ...]
        if explicit is not None:
            candidate_groups = (("explicit_id", (explicit,)),)
        else:
            ids: dict[str, CarbonTransition] = {}
            for annotation in reaction.annotation.values():
                for value in _values(annotation):
                    for entry in self._identifiers.get(value.casefold(), ()):
                        ids[entry.canonical_id] = entry
            alias = next(
                (self._aliases[form] for form in _normal_forms(reaction.id) if form in self._aliases),
                None,
            )
            candidate_groups = (
                ("database_identifier", tuple(ids.values())),
                ("model_alias", (alias,) if alias is not None else ()),
                ("carbon_chemistry", self.transitions),
            )
        for matched_by, candidates in candidate_groups:
            if not candidates:
                continue
            if unknown_carbon:
                if matched_by != "carbon_chemistry":
                    raise MappingError(
                        f"reaction '{reaction.id}' has unrecognized carbon metabolites: {', '.join(unknown_carbon)}"
                    )
                return None
            oriented = []
            for entry in candidates:
                result = self._orient(entry, substrates, products)
                if result is None:
                    continue
                _, substrate_ids, product_ids = result
                if self._compartments_compatible(
                    reaction, entry, substrate_ids, product_ids
                ):
                    oriented.append((entry, result))
            if len(oriented) > 1:
                raise MappingError(
                    f"reaction '{reaction.id}' has ambiguous {matched_by} library matches: "
                    + ", ".join(item.canonical_id for item, _ in oriented)
                )
            if len(oriented) == 1:
                entry, (direction, substrate_ids, product_ids) = oriented[0]
                return ResolvedTransition(
                    reaction.id, entry, direction, matched_by, substrate_ids, product_ids
                )
            if matched_by != "carbon_chemistry":
                raise MappingError(
                    f"reaction '{reaction.id}' {matched_by} matched a transition but its carbon chemistry or direction did not agree"
                )
        return None


@lru_cache(maxsize=1)
def load_default_library() -> TransitionLibrary:
    """Load and validate the package's curated Version-1 data files once."""

    return TransitionLibrary.from_directory(Path(files(__package__) / "data"))


def resolve_model_metadata(
    model: "Model",
    metadata: IsotopeMetadataCollection,
    library: TransitionLibrary | None = None,
) -> ResolvedMetadata:
    """Fill missing atom maps from the library while preserving SBML precedence."""

    transition_library = library or load_default_library()
    reactions = dict(metadata.reactions)
    metabolites = dict(metadata.metabolites)
    records: list[MappingProvenanceRecord] = []
    resolved: list[ResolvedTransition] = []
    for reaction in model.reactions:
        existing = reactions.get(reaction.id)
        if existing is not None:
            records.append(MappingProvenanceRecord(
                reaction.id,
                None,
                "SBML",
                "FLUXEMU_REACTION_METADATA_V1",
                "explicit",
                "metadata-defined" if existing.include_in_isotope_model else "excluded",
                None,
            ))
            continue
        match = transition_library.resolve_reaction(reaction)
        if match is None:
            continue
        generated = export_reaction_metadata(
            match.transition,
            transition_library.metabolites,
            cobra_reaction_id=reaction.id,
            substrate_ids=match.substrate_ids,
            product_ids=match.product_ids,
            direction=match.direction,
        )
        reactions[reaction.id] = generated
        resolved.append(match)
        records.append(MappingProvenanceRecord(
            reaction.id,
            match.transition.canonical_id,
            "library",
            match.transition.provenance.source_identifier,
            match.transition.validation_status,
            match.transition.symmetry,
            "; ".join(match.warnings) or None,
        ))

    for match in resolved:
        for model_id, canonical in zip(match.substrate_ids, (match.transition.products if match.direction == "reverse" else match.transition.substrates)):
            _ensure_metabolite_metadata(model, metabolites, transition_library, model_id, canonical)
        for model_id, canonical in zip(match.product_ids, (match.transition.substrates if match.direction == "reverse" else match.transition.products)):
            _ensure_metabolite_metadata(model, metabolites, transition_library, model_id, canonical)
    return ResolvedMetadata(IsotopeMetadataCollection(reactions, metabolites), tuple(records))


def _ensure_metabolite_metadata(
    model: "Model",
    metadata: dict[str, MetaboliteIsotopeMetadata],
    library: TransitionLibrary,
    model_id: str,
    canonical: str,
) -> None:
    definition = library.metabolites[canonical]
    existing = metadata.get(model_id)
    if existing is None:
        metadata[model_id] = MetaboliteIsotopeMetadata(
            original_cobra_metabolite_id=model_id,
            carbon_count=definition.carbon_count,
            is_carbon_source=False,
            is_excreted=False,
            symmetry=definition.symmetry,
            include_in_isotope_model=True,
        )
        return
    if existing.carbon_count != definition.carbon_count:
        raise MetadataError(
            f"model metabolite '{model_id}' declares {existing.carbon_count} carbons but library '{canonical}' has {definition.carbon_count}"
        )
    if definition.symmetry and not existing.symmetry:
        metadata[model_id] = replace(existing, symmetry=True)


__all__ = [
    "MappingProvenanceRecord", "ResolvedMetadata", "ResolvedTransition", "TransitionLibrary",
    "load_default_library", "resolve_model_metadata",
]
