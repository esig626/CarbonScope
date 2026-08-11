"""Mechanical validation of the canonical scientific contract."""

from __future__ import annotations

import math
from typing import NoReturn

from .schema import (
    AtomPosition,
    AtomTransition,
    CanonicalModel,
    FluxModel,
    IsotopeModel,
    IsotopeParticipant,
    IsotopeReaction,
    MappingBranch,
    PoolQuantity,
    StationaryExperimentSemantics,
    TransientExperimentSemantics,
)

# Mixture weights and tracer fractions are scientific proportions.  This
# tolerance permits only ordinary floating-point summation error.
MIXTURE_ABS_TOLERANCE = 1e-9


class CanonicalModelError(ValueError):
    """The supplied canonical model or stationary experiment is invalid."""


def _fail(message: str) -> NoReturn:
    raise CanonicalModelError(message)


def _nonempty_id(value: object, description: str) -> None:
    if not isinstance(value, str) or not value:
        _fail(f"{description} must be a nonempty string")


def _literal_bool(value: object, description: str) -> None:
    if not isinstance(value, bool):
        _fail(f"{description} must be a literal bool")


def _finite_number(value: object, description: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{description} must be numeric and not bool")
    if not math.isfinite(value):
        _fail(f"{description} must be finite")
    if positive and value <= 0:
        _fail(f"{description} must be positive")


def _unique_ids(records: tuple[object, ...], attribute: str, description: str) -> None:
    seen: set[str] = set()
    for record in records:
        identifier = getattr(record, attribute, None)
        _nonempty_id(identifier, description)
        if identifier in seen:
            _fail(f"duplicate {description}: {identifier!r}")
        seen.add(identifier)


def _tuple(value: object, description: str) -> None:
    if not isinstance(value, tuple):
        _fail(f"{description} must be a tuple")


def validate_flux_model(model: FluxModel) -> None:
    """Validate flux IDs, literal semantics, coefficients, and references."""

    if not isinstance(model, FluxModel):
        _fail("flux model must be a FluxModel")
    _tuple(model.metabolites, "flux metabolites")
    _tuple(model.reactions, "flux reactions")
    _tuple(model.objective.terms, "objective terms")
    _unique_ids(model.metabolites, "metabolite_id", "flux metabolite ID")
    _unique_ids(model.reactions, "reaction_id", "flux reaction ID")
    metabolite_ids = {item.metabolite_id for item in model.metabolites}
    reaction_ids = {item.reaction_id for item in model.reactions}
    for metabolite in model.metabolites:
        _literal_bool(
            metabolite.steady_state_balanced,
            f"steady_state_balanced for {metabolite.metabolite_id!r}",
        )
    for reaction in model.reactions:
        _tuple(reaction.stoichiometric_terms, "stoichiometric terms")
        _finite_number(reaction.lower_bound, f"lower bound for {reaction.reaction_id!r}")
        _finite_number(reaction.upper_bound, f"upper bound for {reaction.reaction_id!r}")
        if reaction.lower_bound > reaction.upper_bound:
            _fail(f"lower bound exceeds upper bound for {reaction.reaction_id!r}")
        for term in reaction.stoichiometric_terms:
            if term.metabolite_id not in metabolite_ids:
                _fail(f"unknown flux metabolite {term.metabolite_id!r}")
            _finite_number(term.coefficient, "stoichiometric coefficient")
    if model.objective.direction not in {"maximise", "minimise"}:
        _fail("objective direction must be exactly 'maximise' or 'minimise'")
    for term in model.objective.terms:
        if term.reaction_id not in reaction_ids:
            _fail(f"objective references unknown reaction {term.reaction_id!r}")
        _finite_number(term.coefficient, "objective coefficient")


def _validate_participant(
    participant: object,
    metabolites: dict[str, object],
    reaction_id: str,
) -> None:
    if not isinstance(participant, IsotopeParticipant):
        _fail(f"malformed isotope participant in reaction {reaction_id!r}")
    _tuple(participant.atom_positions, "isotope participant atom positions")
    metabolite = metabolites.get(participant.metabolite_id)
    if metabolite is None:
        _fail(f"participant references unknown isotope metabolite {participant.metabolite_id!r}")
    if not metabolite.isotope_visible:
        _fail(f"participant metabolite {participant.metabolite_id!r} is not isotope-visible")
    if not participant.atom_positions:
        _fail(f"participant {participant.metabolite_id!r} has no atom positions")
    seen: set[int] = set()
    for position in participant.atom_positions:
        if isinstance(position, bool) or not isinstance(position, int):
            _fail(f"atom position for {participant.metabolite_id!r} must be an integer")
        if position in seen:
            _fail(f"duplicate atom position for participant {participant.metabolite_id!r}")
        if not 1 <= position <= metabolite.carbon_count:
            _fail(f"atom position for {participant.metabolite_id!r} is out of range")
        seen.add(position)


def _declared_atoms(participants: tuple[IsotopeParticipant, ...]) -> tuple[AtomPosition, ...]:
    return tuple(
        AtomPosition(participant.metabolite_id, position)
        for participant in participants
        for position in participant.atom_positions
    )


def _validate_mapping(reaction: IsotopeReaction) -> None:
    expected_sources = _declared_atoms(reaction.substrates)
    expected_destinations = _declared_atoms(reaction.products)
    if len(set(expected_sources)) != len(expected_sources):
        _fail(f"reaction {reaction.reaction_id!r} declares a source atom more than once")
    if len(set(expected_destinations)) != len(expected_destinations):
        _fail(f"reaction {reaction.reaction_id!r} declares a destination atom more than once")
    expected_source_set = set(expected_sources)
    expected_destination_set = set(expected_destinations)
    for branch in reaction.mapping_branches:
        if not isinstance(branch, MappingBranch):
            _fail(f"malformed mapping branch in reaction {reaction.reaction_id!r}")
        _nonempty_id(getattr(branch, "branch_id", None), "mapping branch ID")
        _finite_number(getattr(branch, "weight", None), "mapping branch weight", positive=True)
        transitions = getattr(branch, "transitions", None)
        if not isinstance(transitions, tuple) or not transitions:
            _fail(f"mapping branch {branch.branch_id!r} must contain transitions")
        sources: list[AtomPosition] = []
        destinations: list[AtomPosition] = []
        for transition in transitions:
            if not isinstance(transition, AtomTransition):
                _fail(f"malformed transition in mapping branch {branch.branch_id!r}")
            if not isinstance(transition.source, AtomPosition) or not isinstance(
                transition.destination, AtomPosition
            ):
                _fail(f"malformed atom position in mapping branch {branch.branch_id!r}")
            for atom in (transition.source, transition.destination):
                if not isinstance(atom.metabolite_id, str) or not atom.metabolite_id:
                    _fail(f"malformed atom position in mapping branch {branch.branch_id!r}")
                if isinstance(atom.position, bool) or not isinstance(atom.position, int):
                    _fail(f"malformed atom position in mapping branch {branch.branch_id!r}")
            sources.append(transition.source)
            destinations.append(transition.destination)
        if len(set(sources)) != len(sources):
            _fail(f"duplicate source atom in mapping branch {branch.branch_id!r}")
        if len(set(destinations)) != len(destinations):
            _fail(f"duplicate destination atom in mapping branch {branch.branch_id!r}")
        if set(sources) != expected_source_set:
            _fail(f"mapping branch {branch.branch_id!r} does not contain exactly the declared source atoms")
        if set(destinations) != expected_destination_set:
            _fail(f"mapping branch {branch.branch_id!r} does not contain exactly the declared destination atoms")


def validate_isotope_model(model: IsotopeModel) -> None:
    """Validate isotope declarations and complete explicit atom maps."""

    if not isinstance(model, IsotopeModel):
        _fail("isotope model must be an IsotopeModel")
    _tuple(model.metabolites, "isotope metabolites")
    _tuple(model.reactions, "isotope reactions")
    # Duplicate detection deliberately precedes dictionary construction so a
    # later carbon count can never overwrite the authoritative first record.
    _unique_ids(model.metabolites, "metabolite_id", "isotope metabolite ID")
    _unique_ids(model.reactions, "reaction_id", "isotope reaction ID")
    for metabolite in model.metabolites:
        _literal_bool(metabolite.isotope_visible, "isotope_visible")
        _literal_bool(metabolite.symmetry, "symmetry")
        if isinstance(metabolite.carbon_count, bool) or not isinstance(metabolite.carbon_count, int):
            _fail("carbon_count must be an integer and not bool")
        if metabolite.carbon_count < 0:
            _fail("carbon_count must be nonnegative")
        if metabolite.isotope_visible and metabolite.carbon_count < 1:
            _fail("isotope-visible metabolites must contain at least one carbon")
    metabolites = {item.metabolite_id: item for item in model.metabolites}
    for reaction in model.reactions:
        _tuple(reaction.substrates, "isotope substrates")
        _tuple(reaction.products, "isotope products")
        _tuple(reaction.mapping_branches, "mapping branches")
        _literal_bool(reaction.isotope_enabled, "isotope_enabled")
        if reaction.direction not in {"forward", "reverse"}:
            _fail("isotope direction must be exactly 'forward' or 'reverse'")
        if reaction.directional_id is not None:
            _nonempty_id(reaction.directional_id, "directional isotope reaction ID")
        for participant in reaction.substrates + reaction.products:
            _validate_participant(participant, metabolites, reaction.reaction_id)
        if reaction.mapping_branches:
            _validate_mapping(reaction)
        if reaction.isotope_enabled:
            if not reaction.substrates or not reaction.products:
                _fail(f"enabled isotope reaction {reaction.reaction_id!r} requires substrates and products")
            if not reaction.mapping_branches:
                _fail(f"enabled isotope reaction {reaction.reaction_id!r} requires mapping branches")
            weight = sum(branch.weight for branch in reaction.mapping_branches)
            if not math.isclose(weight, 1.0, rel_tol=0.0, abs_tol=MIXTURE_ABS_TOLERANCE):
                _fail(f"mapping branch weights for {reaction.reaction_id!r} must sum to one")


def validate_canonical_model(model: CanonicalModel) -> None:
    """Validate both engine-independent portions of a canonical model."""

    if not isinstance(model, CanonicalModel):
        _fail("canonical model must be a CanonicalModel")
    validate_flux_model(model.flux_model)
    validate_isotope_model(model.isotope_model)


def _isotope_index(model: CanonicalModel) -> dict[str, object]:
    return {item.metabolite_id: item for item in model.isotope_model.metabolites}


def validate_stationary_experiment(
    model: CanonicalModel,
    experiment: StationaryExperimentSemantics,
) -> None:
    """Validate stationary tracer and target semantics relative to ``model``."""

    validate_canonical_model(model)
    if not isinstance(experiment, StationaryExperimentSemantics):
        _fail("experiment must be StationaryExperimentSemantics")
    _tuple(experiment.tracers, "experiment tracers")
    _tuple(experiment.targets, "experiment targets")
    metabolites = _isotope_index(model)
    for tracer in experiment.tracers:
        _tuple(tracer.isotopomers, "tracer isotopomers")
        metabolite = metabolites.get(tracer.metabolite_id)
        if metabolite is None:
            _fail(f"tracer references unknown isotope metabolite {tracer.metabolite_id!r}")
        if not metabolite.isotope_visible:
            _fail(f"tracer metabolite {tracer.metabolite_id!r} is not isotope-visible")
        if not tracer.isotopomers:
            _fail("tracer isotopomers must be nonempty")
        patterns: set[str] = set()
        total = 0
        for isotopomer in tracer.isotopomers:
            if not isinstance(isotopomer, tuple) or len(isotopomer) != 2:
                _fail("each tracer isotopomer must be exactly (pattern, fraction)")
            pattern, fraction = isotopomer
            if not isinstance(pattern, str) or not pattern.startswith("#"):
                _fail("tracer pattern must begin with '#'")
            labels = pattern[1:]
            if len(labels) != metabolite.carbon_count or any(bit not in "01" for bit in labels):
                _fail("tracer pattern must have one binary character per carbon")
            if pattern in patterns:
                _fail(f"duplicate tracer pattern {pattern!r}")
            patterns.add(pattern)
            _finite_number(fraction, "tracer fraction")
            if not 0 <= fraction <= 1:
                _fail("tracer fraction must be within [0, 1]")
            total += fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=MIXTURE_ABS_TOLERANCE):
            _fail("tracer fractions must sum to one")
        if tracer.correction not in {"yes", "no"}:
            _fail("tracer correction must be exactly 'yes' or 'no'")
    target_ids: set[str] = set()
    for target in experiment.targets:
        _tuple(target.atom_positions, "target atom positions")
        _nonempty_id(target.target_id, "target ID")
        if target.target_id in target_ids:
            _fail(f"duplicate target ID: {target.target_id!r}")
        target_ids.add(target.target_id)
        metabolite = metabolites.get(target.metabolite_id)
        if metabolite is None:
            _fail(f"target references unknown isotope metabolite {target.metabolite_id!r}")
        if not metabolite.isotope_visible:
            _fail(f"target metabolite {target.metabolite_id!r} is not isotope-visible")
        if not target.atom_positions:
            _fail("target atom positions must be nonempty")
        positions: set[int] = set()
        for position in target.atom_positions:
            if isinstance(position, bool) or not isinstance(position, int):
                _fail("target atom position must be an integer and not bool")
            if position in positions:
                _fail("target atom positions must be unique")
            if not 1 <= position <= metabolite.carbon_count:
                _fail("target atom position is out of range")
            positions.add(position)
        _nonempty_id(target.analytical_method, "target analytical method")
        _nonempty_id(target.formula, "target formula")
        if target.correction not in {"yes", "no"}:
            _fail("target correction must be exactly 'yes' or 'no'")


def validate_transient_experiment(
    model: CanonicalModel,
    experiment: TransientExperimentSemantics,
) -> None:
    """Validate V1 transient inputs without changing stationary semantics."""

    validate_canonical_model(model)
    if not isinstance(experiment, TransientExperimentSemantics):
        _fail("experiment must be TransientExperimentSemantics")
    validate_stationary_experiment(
        model, StationaryExperimentSemantics(experiment.tracers, experiment.targets)
    )
    _tuple(experiment.time_points, "transient time points")
    if not experiment.time_points:
        _fail("transient time points must be nonempty")
    previous: float | None = None
    for value in experiment.time_points:
        _finite_number(value, "transient time point")
        numeric = float(value)
        if numeric < 0.0:
            _fail("transient time points must be nonnegative")
        if previous is not None and numeric <= previous:
            _fail("transient time points must be strictly increasing and unique")
        previous = numeric

    _tuple(experiment.pool_quantities, "transient pool quantities")
    known = {item.metabolite_id for item in model.flux_model.metabolites}
    seen: set[str] = set()
    for pool in experiment.pool_quantities:
        if not isinstance(pool, PoolQuantity):
            _fail("transient pool quantities must contain PoolQuantity records")
        _nonempty_id(pool.metabolite_id, "pool-quantity metabolite ID")
        if pool.metabolite_id in seen:
            _fail(f"duplicate pool quantity for metabolite {pool.metabolite_id!r}")
        if pool.metabolite_id not in known:
            _fail(f"pool quantity references unknown metabolite {pool.metabolite_id!r}")
        _finite_number(pool.quantity, f"pool quantity for {pool.metabolite_id!r}", positive=True)
        seen.add(pool.metabolite_id)
    if experiment.initial_internal_mids != "unlabelled":
        _fail("initial_internal_mids must be explicitly 'unlabelled' in transient V1")
