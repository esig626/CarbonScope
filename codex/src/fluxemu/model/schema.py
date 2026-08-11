"""Immutable, engine-independent records for canonical FluxEMU science."""

from __future__ import annotations

from dataclasses import dataclass


Number = int | float


@dataclass(frozen=True, slots=True)
class AtomPosition:
    metabolite_id: str
    position: int


@dataclass(frozen=True, slots=True)
class AtomTransition:
    source: AtomPosition
    destination: AtomPosition


@dataclass(frozen=True, slots=True)
class MappingBranch:
    branch_id: str
    weight: Number
    transitions: tuple[AtomTransition, ...]


@dataclass(frozen=True, slots=True)
class FluxMetabolite:
    metabolite_id: str
    steady_state_balanced: bool


@dataclass(frozen=True, slots=True)
class StoichiometricTerm:
    metabolite_id: str
    coefficient: Number


@dataclass(frozen=True, slots=True)
class FluxReaction:
    reaction_id: str
    stoichiometric_terms: tuple[StoichiometricTerm, ...]
    lower_bound: Number
    upper_bound: Number


@dataclass(frozen=True, slots=True)
class ObjectiveTerm:
    reaction_id: str
    coefficient: Number


@dataclass(frozen=True, slots=True)
class LinearObjective:
    direction: str
    terms: tuple[ObjectiveTerm, ...]


@dataclass(frozen=True, slots=True)
class FluxModel:
    metabolites: tuple[FluxMetabolite, ...]
    reactions: tuple[FluxReaction, ...]
    objective: LinearObjective


@dataclass(frozen=True, slots=True)
class IsotopeMetabolite:
    metabolite_id: str
    carbon_count: int
    isotope_visible: bool
    symmetry: bool


@dataclass(frozen=True, slots=True)
class IsotopeParticipant:
    metabolite_id: str
    atom_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IsotopeReaction:
    reaction_id: str
    direction: str
    isotope_enabled: bool
    substrates: tuple[IsotopeParticipant, ...]
    products: tuple[IsotopeParticipant, ...]
    mapping_branches: tuple[MappingBranch, ...]
    directional_id: str | None = None
    symmetry_semantics: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class IsotopeModel:
    metabolites: tuple[IsotopeMetabolite, ...]
    reactions: tuple[IsotopeReaction, ...]


@dataclass(frozen=True, slots=True)
class Tracer:
    metabolite_id: str
    isotopomers: tuple[tuple[str, Number], ...]
    correction: str


@dataclass(frozen=True, slots=True)
class Target:
    target_id: str
    metabolite_id: str
    atom_positions: tuple[int, ...]
    analytical_method: str
    formula: str
    correction: str


@dataclass(frozen=True, slots=True)
class StationaryExperimentSemantics:
    tracers: tuple[Tracer, ...]
    targets: tuple[Target, ...]


@dataclass(frozen=True, slots=True)
class PoolQuantity:
    """A fixed amount for one balanced dynamic metabolite pool."""

    metabolite_id: str
    quantity: Number


@dataclass(frozen=True, slots=True)
class TransientExperimentSemantics:
    """V1 tracer-step experiment state, separate from stationary semantics."""

    tracers: tuple[Tracer, ...]
    targets: tuple[Target, ...]
    time_points: tuple[Number, ...]
    pool_quantities: tuple[PoolQuantity, ...]
    initial_internal_mids: str


@dataclass(frozen=True, slots=True)
class CanonicalModel:
    flux_model: FluxModel
    isotope_model: IsotopeModel
