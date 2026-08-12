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
class FluxProjectionTerm:
    """One coefficient in an explicit physical-net-flux expression."""

    reaction_id: str
    coefficient: Number


@dataclass(frozen=True, slots=True)
class PhysicalDirectionRef:
    reaction_id: str
    direction: str


@dataclass(frozen=True, slots=True)
class FluxProjectionExpression:
    terms: tuple[FluxProjectionTerm, ...]


@dataclass(frozen=True, slots=True)
class FluxProjectionRule:
    """Serializable map from a complete physical flux vector to one component rate."""

    projection_id: str
    expression: FluxProjectionExpression
    transform: str
    zero_tolerance: Number
    equivalent_expressions: tuple[FluxProjectionExpression, ...] = ()
    covered_physical_directions: tuple[PhysicalDirectionRef, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class DirectionActivity:
    reaction_id: str
    forward_active: bool
    reverse_active: bool
    forward_maximum: Number
    reverse_minimum: Number


@dataclass(frozen=True, slots=True)
class DirectionActivityCertificate:
    certificate_id: str
    zero_tolerance: Number
    activities: tuple[DirectionActivity, ...]
    provenance: tuple[tuple[str, str], ...] = ()


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
    flux_projection: FluxProjectionRule | None = None


@dataclass(frozen=True, slots=True)
class IsotopeModel:
    metabolites: tuple[IsotopeMetabolite, ...]
    reactions: tuple[IsotopeReaction, ...]
    direction_activity_certificate: DirectionActivityCertificate | None = None


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
class ObservationPrecursor:
    metabolite_id: str
    atom_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ObservationTarget:
    target_id: str
    carbon_count: int
    precursors: tuple[ObservationPrecursor, ...]
    provenance: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StationaryExperimentSemantics:
    tracers: tuple[Tracer, ...]
    targets: tuple[Target, ...]
    observation_targets: tuple[ObservationTarget, ...] = ()


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
