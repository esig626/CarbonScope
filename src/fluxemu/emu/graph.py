"""Immutable native EMU graph compiled from canonical atom transitions."""

from __future__ import annotations

from dataclasses import dataclass

from fluxemu.exceptions import MappingError
from fluxemu.model import (
    CanonicalModel,
    FluxProjectionRule,
    StationaryExperimentSemantics,
    experiment_fingerprint,
    model_fingerprint,
    validate_canonical_model,
    validate_stationary_experiment,
)


@dataclass(frozen=True, slots=True)
class EMU:
    """An isotope pool fragment; positions retain declared atom order."""

    metabolite_id: str
    atom_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.metabolite_id or not self.atom_positions:
            raise MappingError("an EMU requires a metabolite ID and atom positions")
        if len(set(self.atom_positions)) != len(self.atom_positions):
            raise MappingError(f"EMU {self.metabolite_id!r} contains duplicate atom positions")

    @property
    def size(self) -> int:
        return len(self.atom_positions)


@dataclass(frozen=True, slots=True)
class EMUContribution:
    reaction_id: str
    branch_id: str
    branch_weight: float
    product: EMU
    precursors: tuple[EMU, ...]
    flux_projection: FluxProjectionRule | None = None


@dataclass(frozen=True, slots=True)
class EMULayer:
    size: int
    unknowns: tuple[EMU, ...]


@dataclass(frozen=True, slots=True)
class CompiledEMUPlan:
    model: CanonicalModel
    experiment: StationaryExperimentSemantics
    model_fingerprint: str
    experiment_fingerprint: str
    targets: tuple[tuple[str, EMU], ...]
    emus: tuple[EMU, ...]
    contributions: tuple[EMUContribution, ...]
    layers: tuple[EMULayer, ...]
    source_emus: tuple[EMU, ...]
    observations: tuple[tuple[str, tuple[EMU, ...]], ...] = ()


def _physical_direction(reaction) -> int:
    """Return +1 for forward-only and -1 for reverse-only canonical flux."""

    lower, upper = float(reaction.lower_bound), float(reaction.upper_bound)
    if lower >= 0.0:
        return 1
    if upper <= 0.0:
        return -1
    raise MappingError(
        f"reaction {reaction.reaction_id!r} has ambiguous bidirectional bounds; "
        "native V1 stationary execution requires one physical direction"
    )


def compile_emu_plan(
    model: CanonicalModel, experiment: StationaryExperimentSemantics
) -> CompiledEMUPlan:
    """Trace only target-required EMUs backwards through explicit mappings."""

    validate_canonical_model(model)
    validate_stationary_experiment(model, experiment)
    if any(item.correction != "no" for item in experiment.tracers + experiment.targets):
        raise MappingError("native stationary EMU supports only explicit no-correction semantics")

    flux_by_id = {item.reaction_id: item for item in model.flux_model.reactions}
    balanced = {item.metabolite_id for item in model.flux_model.metabolites if item.steady_state_balanced}
    sources = {item.metabolite_id for item in experiment.tracers}
    targets = tuple(
        (target.target_id, EMU(target.metabolite_id, target.atom_positions))
        for target in experiment.targets
    )
    observations = tuple(
        (item.target_id, tuple(EMU(p.metabolite_id, p.atom_positions) for p in item.precursors))
        for item in experiment.observation_targets
    )

    ordered_emus: list[EMU] = []
    contributions: list[EMUContribution] = []
    visited: set[EMU] = set()

    def trace(emu: EMU) -> None:
        if emu in visited:
            return
        visited.add(emu)
        ordered_emus.append(emu)
        if emu.metabolite_id in sources:
            return
        if emu.metabolite_id not in balanced and emu not in {item[1] for item in targets}:
            raise MappingError(
                f"unbalanced isotope substrate {emu.metabolite_id!r} is not a declared tracer"
            )

        found = False
        for reaction in model.isotope_model.reactions:
            if not reaction.isotope_enabled:
                continue
            product = next((p for p in reaction.products if p.metabolite_id == emu.metabolite_id), None)
            if product is None:
                continue
            if reaction.flux_projection is None:
                physical = flux_by_id.get(reaction.reaction_id)
                if physical is None:
                    raise MappingError(f"isotope reaction {reaction.reaction_id!r} has no flux reaction")
                direction = _physical_direction(physical)
                expected = 1 if reaction.direction == "forward" else -1
                if direction != expected:
                    raise MappingError(
                        f"isotope direction for {reaction.reaction_id!r} conflicts with canonical bounds"
                    )
            for branch in reaction.mapping_branches:
                selected = []
                for destination_position in emu.atom_positions:
                    matches = tuple(
                        transition.source
                        for transition in branch.transitions
                        if transition.destination.metabolite_id == emu.metabolite_id
                        and transition.destination.position == destination_position
                    )
                    if len(matches) != 1:
                        raise MappingError(
                            f"branch {branch.branch_id!r} does not explicitly map target atom "
                            f"{emu.metabolite_id}:{destination_position}"
                        )
                    selected.append(matches[0])
                precursor_list = []
                for participant in reaction.substrates:
                    positions = tuple(
                        atom.position for atom in selected if atom.metabolite_id == participant.metabolite_id
                    )
                    if positions:
                        precursor_list.append(EMU(participant.metabolite_id, positions))
                if sum(item.size for item in precursor_list) != emu.size:
                    raise MappingError(
                        f"branch {branch.branch_id!r} cannot supply every atom of {emu.metabolite_id!r}"
                    )
                contribution = EMUContribution(
                    reaction.reaction_id,
                    branch.branch_id,
                    float(branch.weight),
                    emu,
                    tuple(precursor_list),
                    reaction.flux_projection,
                )
                contributions.append(contribution)
                found = True
                for precursor in precursor_list:
                    trace(precursor)
        if not found:
            raise MappingError(f"no explicit production mapping supplies required EMU {emu!r}")

    for _, target_emu in targets:
        trace(target_emu)
    for _, precursor_emus in observations:
        for precursor_emu in precursor_emus:
            trace(precursor_emu)

    # Any physical producer of a required non-source pool needs an explicit map.
    planned_metabolites = tuple(dict.fromkeys(
        item.metabolite_id for item in ordered_emus if item.metabolite_id not in sources
    ))
    for metabolite_id in planned_metabolites:
        covered = {
            (reference.reaction_id, reference.direction)
            for item in model.isotope_model.reactions
            if item.isotope_enabled and any(p.metabolite_id == metabolite_id for p in item.products)
            for reference in (
                item.flux_projection.covered_physical_directions
                if item.flux_projection is not None
                else ()
            )
        }
        for reaction in model.flux_model.reactions:
            coefficient = sum(
                float(term.coefficient)
                for term in reaction.stoichiometric_terms
                if term.metabolite_id == metabolite_id
            )
            activity = None
            certificate = model.isotope_model.direction_activity_certificate
            if certificate is not None:
                activity = next(item for item in certificate.activities if item.reaction_id == reaction.reaction_id)
            forward_possible = activity.forward_active if activity is not None else float(reaction.upper_bound) > 0
            reverse_possible = activity.reverse_active if activity is not None else float(reaction.lower_bound) < 0
            possible = []
            if forward_possible and coefficient > 0:
                possible.append("forward")
            if reverse_possible and coefficient < 0:
                possible.append("reverse")
            for direction in possible:
                legacy_mapped = any(
                    item.flux_projection is None and item.reaction_id == reaction.reaction_id
                    and any(p.metabolite_id == metabolite_id for p in item.products)
                    for item in model.isotope_model.reactions
                )
                if not legacy_mapped and (reaction.reaction_id, direction) not in covered:
                    raise MappingError(
                        f"flux reaction {reaction.reaction_id!r} {direction} produces required pool "
                        f"{metabolite_id!r} without an explicit isotope mapping or directional coverage"
                    )

    unknowns = tuple(
        item for item in ordered_emus if item.metabolite_id in balanced and item.metabolite_id not in sources
    )
    sizes = sorted({item.size for item in unknowns})
    layers = tuple(EMULayer(size, tuple(item for item in unknowns if item.size == size)) for size in sizes)
    return CompiledEMUPlan(
        model,
        experiment,
        model_fingerprint(model),
        experiment_fingerprint(model, experiment),
        targets,
        tuple(ordered_emus),
        tuple(contributions),
        layers,
        tuple(item for item in ordered_emus if item.metabolite_id in sources),
        observations,
    )


__all__ = ["CompiledEMUPlan", "EMU", "EMUContribution", "EMULayer", "compile_emu_plan"]
