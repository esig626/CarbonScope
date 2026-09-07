"""Strict normalized schema for curated carbon atom transitions.

The on-disk YAML deliberately contains ``metabolite.C<n>`` references rather
than the letter strings expected by mfapy.  This module is the only parser for
that format and rejects unknown fields, incomplete provenance, and ambiguous
atom references before an entry can be used by the forward pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from fluxemu.exceptions import CarbonTransitionValidationError


ValidationStatus = Literal["gold", "curated", "provisional"]
ReverseAtomMap = Literal["derived", "not_applicable"] | tuple["AtomTransition", ...]


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CarbonTransitionValidationError(f"{context} must be a mapping with string keys")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CarbonTransitionValidationError(f"{context} must be a non-empty string")
    return value.strip()


def _sequence(value: Any, context: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise CarbonTransitionValidationError(f"{context} must be a list")
    try:
        return tuple(value)
    except TypeError as error:
        raise CarbonTransitionValidationError(f"{context} must be a list") from error


def _check_fields(
    data: Mapping[str, Any], *, required: set[str], allowed: set[str], context: str
) -> None:
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - allowed)
    if missing:
        raise CarbonTransitionValidationError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )
    if unknown:
        raise CarbonTransitionValidationError(
            f"{context} contains unknown field(s): {', '.join(unknown)}"
        )


def _text_tuple(value: Any, context: str) -> tuple[str, ...]:
    return tuple(_text(item, context) for item in _sequence(value, context))


def _identifier_mapping(value: Any, context: str) -> Mapping[str, tuple[str, ...]]:
    data = _mapping(value, context)
    return MappingProxyType(
        {key: _text_tuple(item, f"{context}.{key}") for key, item in data.items()}
    )


@dataclass(frozen=True, order=True)
class AtomRef:
    """One explicit carbon position in a canonical metabolite."""

    metabolite: str
    position: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "metabolite", _text(self.metabolite, "atom.metabolite"))
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position < 1:
            raise CarbonTransitionValidationError("atom.position must be a positive integer")

    @classmethod
    def parse(cls, value: Any, context: str) -> "AtomRef":
        text = _text(value, context)
        metabolite, separator, suffix = text.rpartition(".C")
        if not separator or not metabolite or not suffix.isdecimal() or suffix.startswith("0"):
            raise CarbonTransitionValidationError(
                f"{context} must be formatted as 'metabolite.C<n>'"
            )
        return cls(metabolite, int(suffix))

    def __str__(self) -> str:
        return f"{self.metabolite}.C{self.position}"


@dataclass(frozen=True)
class AtomTransition:
    """A directed carbon correspondence from one substrate to one product."""

    source: AtomRef
    destination: AtomRef

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "AtomTransition":
        data = _mapping(value, context)
        _check_fields(data, required={"from", "to"}, allowed={"from", "to"}, context=context)
        return cls(AtomRef.parse(data["from"], f"{context}.from"), AtomRef.parse(data["to"], f"{context}.to"))

    def inverted(self) -> "AtomTransition":
        return AtomTransition(self.destination, self.source)

    def to_dict(self) -> dict[str, str]:
        return {"from": str(self.source), "to": str(self.destination)}


def _atom_map(value: Any, context: str) -> tuple[AtomTransition, ...]:
    result = tuple(
        AtomTransition.from_dict(item, f"{context}[{index}]")
        for index, item in enumerate(_sequence(value, context))
    )
    if not result:
        raise CarbonTransitionValidationError(f"{context} must contain at least one transition")
    return result


@dataclass(frozen=True)
class MappingBranch:
    """One explicitly weighted, chemically equivalent atom-map branch."""

    branch_id: str
    weight: float
    atom_map: tuple[AtomTransition, ...]

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "MappingBranch":
        data = _mapping(value, context)
        _check_fields(data, required={"id", "weight", "atom_map"}, allowed={"id", "weight", "atom_map"}, context=context)
        weight = data["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0.0 < float(weight) <= 1.0:
            raise CarbonTransitionValidationError(f"{context}.weight must be in (0, 1]")
        return cls(_text(data["id"], f"{context}.id"), float(weight), _atom_map(data["atom_map"], f"{context}.atom_map"))

    def inverted(self) -> "MappingBranch":
        return MappingBranch(self.branch_id, self.weight, tuple(item.inverted() for item in self.atom_map))


@dataclass(frozen=True)
class Provenance:
    """Exact source metadata required for every library reaction."""

    source_type: str
    citation: str
    source_identifier: str
    source_location: str
    notes: str

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "Provenance":
        data = _mapping(value, context)
        fields = {"source_type", "citation", "source_identifier", "source_location", "notes"}
        _check_fields(data, required=fields, allowed=fields, context=context)
        return cls(*(_text(data[field], f"{context}.{field}") for field in (
            "source_type", "citation", "source_identifier", "source_location", "notes"
        )))

    def to_dict(self) -> dict[str, str]:
        return {
            "source_type": self.source_type,
            "citation": self.citation,
            "source_identifier": self.source_identifier,
            "source_location": self.source_location,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MetaboliteDefinition:
    """One globally numbered central-carbon metabolite."""

    canonical_id: str
    name: str
    carbon_count: int
    carbon_positions: tuple[str, ...]
    synonyms: tuple[str, ...]
    identifiers: Mapping[str, tuple[str, ...]]
    numbering_source: str
    stereochemistry_notes: str
    symmetry: bool

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "MetaboliteDefinition":
        data = _mapping(value, context)
        fields = {
            "canonical_id", "name", "carbon_count", "carbon_positions", "synonyms",
            "identifiers", "numbering_source", "stereochemistry_notes", "symmetry",
        }
        _check_fields(data, required=fields, allowed=fields, context=context)
        count = data["carbon_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise CarbonTransitionValidationError(f"{context}.carbon_count must be a positive integer")
        positions = _text_tuple(data["carbon_positions"], f"{context}.carbon_positions")
        expected = tuple(f"C{number}" for number in range(1, count + 1))
        if positions != expected:
            raise CarbonTransitionValidationError(
                f"{context}.carbon_positions must be {list(expected)!r} in canonical order"
            )
        symmetry = data["symmetry"]
        if not isinstance(symmetry, bool):
            raise CarbonTransitionValidationError(f"{context}.symmetry must be boolean")
        return cls(
            _text(data["canonical_id"], f"{context}.canonical_id"),
            _text(data["name"], f"{context}.name"),
            count,
            positions,
            _text_tuple(data["synonyms"], f"{context}.synonyms"),
            _identifier_mapping(data["identifiers"], f"{context}.identifiers"),
            _text(data["numbering_source"], f"{context}.numbering_source"),
            _text(data["stereochemistry_notes"], f"{context}.stereochemistry_notes"),
            symmetry,
        )


@dataclass(frozen=True)
class CarbonTransition:
    """A canonical, source-audited biochemical carbon transition."""

    canonical_id: str
    name: str
    pathway: str
    substrates: tuple[str, ...]
    products: tuple[str, ...]
    carbon_counts: Mapping[str, int]
    forward_atom_map: tuple[AtomTransition, ...]
    reverse_atom_map: ReverseAtomMap
    mapping_branches: tuple[MappingBranch, ...]
    reversible: bool
    symmetry: str
    stereochemistry_notes: str
    numbering_convention: str
    aliases: tuple[str, ...]
    ec_numbers: tuple[str, ...]
    identifiers: Mapping[str, tuple[str, ...]]
    provenance: Provenance
    validation_status: ValidationStatus
    comments: str

    @classmethod
    def from_dict(cls, value: Any, context: str) -> "CarbonTransition":
        data = _mapping(value, context)
        fields = {
            "canonical_id", "name", "pathway", "substrates", "products", "carbon_counts",
            "forward_atom_map", "reverse_atom_map", "mapping_branches", "reversible", "symmetry",
            "stereochemistry_notes", "numbering_convention", "aliases", "ec_numbers", "identifiers",
            "provenance", "validation_status", "comments",
        }
        _check_fields(data, required=fields, allowed=fields, context=context)
        counts_raw = _mapping(data["carbon_counts"], f"{context}.carbon_counts")
        counts: dict[str, int] = {}
        for metabolite, count in counts_raw.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise CarbonTransitionValidationError(
                    f"{context}.carbon_counts.{metabolite} must be a positive integer"
                )
            counts[_text(metabolite, f"{context}.carbon_counts key")] = count
        reversible = data["reversible"]
        if not isinstance(reversible, bool):
            raise CarbonTransitionValidationError(f"{context}.reversible must be boolean")
        reverse_raw = data["reverse_atom_map"]
        if isinstance(reverse_raw, str) and reverse_raw in {"derived", "not_applicable"}:
            reverse: ReverseAtomMap = reverse_raw
        else:
            reverse = _atom_map(reverse_raw, f"{context}.reverse_atom_map")
        status = _text(data["validation_status"], f"{context}.validation_status")
        if status not in {"gold", "curated", "provisional"}:
            raise CarbonTransitionValidationError(
                f"{context}.validation_status must be gold, curated, or provisional"
            )
        branches = tuple(
            MappingBranch.from_dict(item, f"{context}.mapping_branches[{index}]")
            for index, item in enumerate(_sequence(data["mapping_branches"], f"{context}.mapping_branches"))
        )
        return cls(
            _text(data["canonical_id"], f"{context}.canonical_id"),
            _text(data["name"], f"{context}.name"),
            _text(data["pathway"], f"{context}.pathway"),
            _text_tuple(data["substrates"], f"{context}.substrates"),
            _text_tuple(data["products"], f"{context}.products"),
            MappingProxyType(counts),
            _atom_map(data["forward_atom_map"], f"{context}.forward_atom_map"),
            reverse,
            branches,
            reversible,
            _text(data["symmetry"], f"{context}.symmetry"),
            _text(data["stereochemistry_notes"], f"{context}.stereochemistry_notes"),
            _text(data["numbering_convention"], f"{context}.numbering_convention"),
            _text_tuple(data["aliases"], f"{context}.aliases"),
            _text_tuple(data["ec_numbers"], f"{context}.ec_numbers"),
            _identifier_mapping(data["identifiers"], f"{context}.identifiers"),
            Provenance.from_dict(data["provenance"], f"{context}.provenance"),
            status,  # type: ignore[arg-type]
            _text(data["comments"], f"{context}.comments"),
        )

    @property
    def forward_branches(self) -> tuple[MappingBranch, ...]:
        return self.mapping_branches or (MappingBranch("primary", 1.0, self.forward_atom_map),)

    @property
    def reverse_branches(self) -> tuple[MappingBranch, ...]:
        if not self.reversible:
            raise CarbonTransitionValidationError(
                f"{self.canonical_id} is irreversible and has no reverse atom map"
            )
        if self.reverse_atom_map == "derived":
            return tuple(branch.inverted() for branch in self.forward_branches)
        if self.reverse_atom_map == "not_applicable":
            raise CarbonTransitionValidationError(
                f"{self.canonical_id} is reversible but declares no reverse atom map"
            )
        return (MappingBranch("reverse", 1.0, self.reverse_atom_map),)

    def branch_for_direction(self, direction: Literal["forward", "reverse"]) -> tuple[MappingBranch, ...]:
        return self.forward_branches if direction == "forward" else self.reverse_branches


__all__ = [
    "AtomRef", "AtomTransition", "CarbonTransition", "MappingBranch", "MetaboliteDefinition",
    "Provenance", "ValidationStatus",
]
