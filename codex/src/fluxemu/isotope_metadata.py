"""Validated isotope metadata stored as canonical JSON in SBML notes.

COBRApy's SBML notes writer interpolates note values directly into XHTML.
Consequently the canonical JSON is HTML-escaped before it enters ``notes``
and unescaped before JSON decoding. This preserves XML-sensitive identifiers
without using annotations as an application-specific data container.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape, unescape
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Mapping, TYPE_CHECKING

from .exceptions import MetadataError


if TYPE_CHECKING:
    from cobra import Metabolite, Model, Reaction


REACTION_METADATA_NOTE_KEY = "FLUXEMU_REACTION_METADATA_V1"
METABOLITE_METADATA_NOTE_KEY = "FLUXEMU_METABOLITE_METADATA_V1"
METADATA_SCHEMA_VERSION = 1
Direction = Literal["forward", "reverse"]


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(f"{context} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise MetadataError(f"{context} must be true or false")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MetadataError(f"{context} must be an integer of at least {minimum}")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetadataError(f"{context} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise MetadataError(f"{context} keys must be strings")
    return value


def _check_keys(
    data: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise MetadataError(
            f"{context} contains unknown field(s): {', '.join(unknown)}"
        )
    missing = sorted(required - set(data))
    if missing:
        raise MetadataError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _tuple(value: Any, context: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise MetadataError(f"{context} must be a list")
    try:
        return tuple(value)
    except TypeError as error:
        raise MetadataError(f"{context} must be a list") from error


@dataclass(frozen=True)
class AtomMappedParticipant:
    """One tracked reaction participant with atoms in metabolic order.

    Matching atom labels on the substrate and product sides define atom
    correspondence. Tuple order defines the carbon order used by mfapy.
    """

    metabolite_id: str
    atom_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metabolite_id",
            _text(self.metabolite_id, "participant.metabolite_id"),
        )
        labels = _tuple(self.atom_labels, "participant.atom_labels")
        if not labels:
            raise MetadataError("participant.atom_labels must not be empty")
        normalized: list[str] = []
        for label in labels:
            atom_label = _text(label, "atom label")
            if any(character.isspace() for character in atom_label):
                raise MetadataError(f"atom label '{atom_label}' contains whitespace")
            if "+" in atom_label or "-->" in atom_label:
                raise MetadataError(
                    f"atom label '{atom_label}' contains an mfapy separator"
                )
            normalized.append(atom_label)
        if len(set(normalized)) != len(normalized):
            raise MetadataError(
                f"participant '{self.metabolite_id}' contains duplicate atom labels"
            )
        object.__setattr__(self, "atom_labels", tuple(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metabolite_id": self.metabolite_id,
            "atom_labels": list(self.atom_labels),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "AtomMappedParticipant":
        data = _mapping(value, "reaction participant")
        fields = {"metabolite_id", "atom_labels"}
        _check_keys(
            data,
            allowed=fields,
            required=fields,
            context="reaction participant",
        )
        return cls(
            metabolite_id=data["metabolite_id"],
            atom_labels=_tuple(data["atom_labels"], "participant.atom_labels"),
        )


@dataclass(frozen=True)
class ReactionIsotopeMetadata:
    """Directional atom-transition metadata for one COBRA reaction."""

    original_cobra_reaction_id: str
    directional_id: str
    direction: Direction
    include_in_isotope_model: bool
    substrates: tuple[AtomMappedParticipant, ...]
    products: tuple[AtomMappedParticipant, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_cobra_reaction_id",
            _text(
                self.original_cobra_reaction_id,
                "reaction.original_cobra_reaction_id",
            ),
        )
        object.__setattr__(
            self,
            "directional_id",
            _text(self.directional_id, "reaction.directional_id"),
        )
        direction = _text(self.direction, "reaction.direction").lower()
        if direction not in {"forward", "reverse"}:
            raise MetadataError("reaction.direction must be 'forward' or 'reverse'")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(
            self,
            "include_in_isotope_model",
            _boolean(
                self.include_in_isotope_model,
                "reaction.include_in_isotope_model",
            ),
        )

        substrates = _tuple(self.substrates, "reaction.substrates")
        products = _tuple(self.products, "reaction.products")
        if not all(isinstance(item, AtomMappedParticipant) for item in substrates):
            raise MetadataError("reaction.substrates contains an invalid participant")
        if not all(isinstance(item, AtomMappedParticipant) for item in products):
            raise MetadataError("reaction.products contains an invalid participant")
        if self.include_in_isotope_model and not (substrates or products):
            raise MetadataError(
                "an included isotope reaction must track a substrate or product"
            )
        self._validate_side(substrates, "substrates")
        self._validate_side(products, "products")
        object.__setattr__(self, "substrates", substrates)
        object.__setattr__(self, "products", products)

    @staticmethod
    def _validate_side(
        participants: tuple[AtomMappedParticipant, ...], side: str
    ) -> None:
        metabolite_ids = [item.metabolite_id for item in participants]
        if len(set(metabolite_ids)) != len(metabolite_ids):
            raise MetadataError(
                f"reaction.{side} contains a duplicate metabolite ID"
            )
        labels = [label for item in participants for label in item.atom_labels]
        if len(set(labels)) != len(labels):
            raise MetadataError(
                f"reaction.{side} contains an ambiguous duplicate atom label"
            )

    @property
    def substrate_atom_labels(self) -> tuple[str, ...]:
        return tuple(
            label for participant in self.substrates for label in participant.atom_labels
        )

    @property
    def product_atom_labels(self) -> tuple[str, ...]:
        return tuple(
            label for participant in self.products for label in participant.atom_labels
        )

    @property
    def conserved_atom_labels(self) -> tuple[str, ...]:
        products = set(self.product_atom_labels)
        return tuple(label for label in self.substrate_atom_labels if label in products)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": METADATA_SCHEMA_VERSION,
            "kind": "reaction",
            "original_cobra_reaction_id": self.original_cobra_reaction_id,
            "directional_id": self.directional_id,
            "direction": self.direction,
            "include_in_isotope_model": self.include_in_isotope_model,
            "substrates": [item.to_dict() for item in self.substrates],
            "products": [item.to_dict() for item in self.products],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReactionIsotopeMetadata":
        data = _mapping(value, "reaction isotope metadata")
        fields = {
            "schema_version",
            "kind",
            "original_cobra_reaction_id",
            "directional_id",
            "direction",
            "include_in_isotope_model",
            "substrates",
            "products",
        }
        _check_keys(
            data,
            allowed=fields,
            required=fields,
            context="reaction isotope metadata",
        )
        _validate_header(data, "reaction")
        return cls(
            original_cobra_reaction_id=data["original_cobra_reaction_id"],
            directional_id=data["directional_id"],
            direction=data["direction"],
            include_in_isotope_model=data["include_in_isotope_model"],
            substrates=tuple(
                AtomMappedParticipant.from_dict(item)
                for item in _tuple(data["substrates"], "reaction.substrates")
            ),
            products=tuple(
                AtomMappedParticipant.from_dict(item)
                for item in _tuple(data["products"], "reaction.products")
            ),
        )


@dataclass(frozen=True)
class MetaboliteIsotopeMetadata:
    """Carbon tracking properties for one COBRA metabolite."""

    original_cobra_metabolite_id: str
    carbon_count: int
    is_carbon_source: bool
    is_excreted: bool
    symmetry: bool
    include_in_isotope_model: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_cobra_metabolite_id",
            _text(
                self.original_cobra_metabolite_id,
                "metabolite.original_cobra_metabolite_id",
            ),
        )
        object.__setattr__(
            self,
            "carbon_count",
            _integer(self.carbon_count, "metabolite.carbon_count", minimum=0),
        )
        for name in (
            "is_carbon_source",
            "is_excreted",
            "symmetry",
            "include_in_isotope_model",
        ):
            object.__setattr__(
                self,
                name,
                _boolean(getattr(self, name), f"metabolite.{name}"),
            )
        if self.include_in_isotope_model and self.carbon_count == 0:
            raise MetadataError(
                "an included isotope metabolite must have at least one carbon"
            )
        if (self.is_carbon_source or self.is_excreted) and not self.include_in_isotope_model:
            raise MetadataError(
                "a carbon source or excreted metabolite must be included in the isotope model"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": METADATA_SCHEMA_VERSION,
            "kind": "metabolite",
            "original_cobra_metabolite_id": self.original_cobra_metabolite_id,
            "carbon_count": self.carbon_count,
            "is_carbon_source": self.is_carbon_source,
            "is_excreted": self.is_excreted,
            "symmetry": self.symmetry,
            "include_in_isotope_model": self.include_in_isotope_model,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MetaboliteIsotopeMetadata":
        data = _mapping(value, "metabolite isotope metadata")
        fields = {
            "schema_version",
            "kind",
            "original_cobra_metabolite_id",
            "carbon_count",
            "is_carbon_source",
            "is_excreted",
            "symmetry",
            "include_in_isotope_model",
        }
        _check_keys(
            data,
            allowed=fields,
            required=fields,
            context="metabolite isotope metadata",
        )
        _validate_header(data, "metabolite")
        return cls(
            original_cobra_metabolite_id=data["original_cobra_metabolite_id"],
            carbon_count=data["carbon_count"],
            is_carbon_source=data["is_carbon_source"],
            is_excreted=data["is_excreted"],
            symmetry=data["symmetry"],
            include_in_isotope_model=data["include_in_isotope_model"],
        )


def _validate_header(data: Mapping[str, Any], expected_kind: str) -> None:
    version = _integer(data.get("schema_version"), "metadata.schema_version", minimum=1)
    if version != METADATA_SCHEMA_VERSION:
        raise MetadataError(f"unsupported isotope metadata schema_version {version}")
    if data.get("kind") != expected_kind:
        raise MetadataError(
            f"expected {expected_kind} isotope metadata, found {data.get('kind')!r}"
        )


MetadataValue = ReactionIsotopeMetadata | MetaboliteIsotopeMetadata | Mapping[str, Any]


def encode_metadata_note(value: MetadataValue) -> str:
    """Return canonical, XHTML-safe JSON for a COBRApy notes value."""

    if isinstance(value, (ReactionIsotopeMetadata, MetaboliteIsotopeMetadata)):
        payload: Mapping[str, Any] = value.to_dict()
    else:
        payload = _mapping(value, "isotope metadata")
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise MetadataError(f"isotope metadata is not valid JSON: {error}") from error
    return escape(canonical, quote=True)


def decode_metadata_note(value: Any) -> dict[str, Any]:
    """Decode an XHTML-safe COBRApy notes value into a JSON object."""

    if not isinstance(value, str) or not value:
        raise MetadataError("isotope metadata note must be a non-empty string")
    try:
        payload = json.loads(unescape(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise MetadataError(f"isotope metadata note is not valid JSON: {error}") from error
    return dict(_mapping(payload, "isotope metadata note"))


def set_reaction_metadata(
    reaction: "Reaction", metadata: ReactionIsotopeMetadata
) -> None:
    """Attach validated reaction metadata to a COBRA reaction's notes."""

    if not isinstance(metadata, ReactionIsotopeMetadata):
        raise MetadataError("reaction metadata has the wrong type")
    if reaction.id != metadata.original_cobra_reaction_id:
        raise MetadataError(
            f"reaction metadata identifies '{metadata.original_cobra_reaction_id}', "
            f"but is attached to '{reaction.id}'"
        )
    reaction.notes[REACTION_METADATA_NOTE_KEY] = encode_metadata_note(metadata)


def get_reaction_metadata(
    reaction: "Reaction", *, required: bool = False
) -> ReactionIsotopeMetadata | None:
    """Read and validate reaction isotope metadata from COBRA notes."""

    encoded = reaction.notes.get(REACTION_METADATA_NOTE_KEY)
    if encoded is None:
        if required:
            raise MetadataError(
                f"reaction '{reaction.id}' is missing isotope metadata"
            )
        return None
    metadata = ReactionIsotopeMetadata.from_dict(decode_metadata_note(encoded))
    if metadata.original_cobra_reaction_id != reaction.id:
        raise MetadataError(
            f"reaction '{reaction.id}' contains metadata for "
            f"'{metadata.original_cobra_reaction_id}'"
        )
    return metadata


def set_metabolite_metadata(
    metabolite: "Metabolite", metadata: MetaboliteIsotopeMetadata
) -> None:
    """Attach validated metabolite metadata to a COBRA metabolite's notes."""

    if not isinstance(metadata, MetaboliteIsotopeMetadata):
        raise MetadataError("metabolite metadata has the wrong type")
    if metabolite.id != metadata.original_cobra_metabolite_id:
        raise MetadataError(
            f"metabolite metadata identifies '{metadata.original_cobra_metabolite_id}', "
            f"but is attached to '{metabolite.id}'"
        )
    metabolite.notes[METABOLITE_METADATA_NOTE_KEY] = encode_metadata_note(metadata)


def get_metabolite_metadata(
    metabolite: "Metabolite", *, required: bool = False
) -> MetaboliteIsotopeMetadata | None:
    """Read and validate metabolite isotope metadata from COBRA notes."""

    encoded = metabolite.notes.get(METABOLITE_METADATA_NOTE_KEY)
    if encoded is None:
        if required:
            raise MetadataError(
                f"metabolite '{metabolite.id}' is missing isotope metadata"
            )
        return None
    metadata = MetaboliteIsotopeMetadata.from_dict(decode_metadata_note(encoded))
    if metadata.original_cobra_metabolite_id != metabolite.id:
        raise MetadataError(
            f"metabolite '{metabolite.id}' contains metadata for "
            f"'{metadata.original_cobra_metabolite_id}'"
        )
    return metadata


@dataclass(frozen=True)
class IsotopeMetadataCollection:
    """All decoded isotope annotations found on a COBRA model."""

    reactions: Mapping[str, ReactionIsotopeMetadata]
    metabolites: Mapping[str, MetaboliteIsotopeMetadata]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reactions", MappingProxyType(dict(self.reactions)))
        object.__setattr__(
            self, "metabolites", MappingProxyType(dict(self.metabolites))
        )

    @property
    def included_reactions(self) -> tuple[ReactionIsotopeMetadata, ...]:
        return tuple(
            metadata
            for metadata in self.reactions.values()
            if metadata.include_in_isotope_model
        )

    @property
    def included_metabolites(self) -> tuple[MetaboliteIsotopeMetadata, ...]:
        return tuple(
            metadata
            for metadata in self.metabolites.values()
            if metadata.include_in_isotope_model
        )


def collect_isotope_metadata(model: "Model") -> IsotopeMetadataCollection:
    """Decode every FluxEMU metadata note present on a COBRA model."""

    reaction_metadata: dict[str, ReactionIsotopeMetadata] = {}
    for reaction in model.reactions:
        metadata = get_reaction_metadata(reaction)
        if metadata is not None:
            reaction_metadata[reaction.id] = metadata

    metabolite_metadata: dict[str, MetaboliteIsotopeMetadata] = {}
    for metabolite in model.metabolites:
        metadata = get_metabolite_metadata(metabolite)
        if metadata is not None:
            metabolite_metadata[metabolite.id] = metadata

    return IsotopeMetadataCollection(
        reactions=reaction_metadata,
        metabolites=metabolite_metadata,
    )
